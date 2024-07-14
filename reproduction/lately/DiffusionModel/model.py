import torch
import torch.nn as nn
from tqdm import tqdm

from lossFunc.WeightedLoss import *
from interpret import *

import enum


def extract(a, t, x_shape):
    """
    将 a 按照 t 进行切片，并将切片后的结果展平为 shape=[batch_size, 1, 1, 1, ...]
    Args:
        a (torch.Tensor):
        t (torch.Tensor | int):
        x_shape (tuple|torch.Size):
    """
    if isinstance(t, torch.Tensor):
        out = a.gather(dim=-1, index=t)
        return out.view(x_shape[0], *((1,) * (len(x_shape) - 1)))
    elif isinstance(t, int):
        out = a[t]
        return out.repeat(x_shape[0], *((1,) * (len(x_shape) - 1)))
    else:
        raise ValueError("t must be int or tensor")


class SamplingType(enum.Enum):
    STOCHASTIC = enum.auto()
    ANTITHETIC = enum.auto()


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, t_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.t_dim = t_dim

    def forward(self, x):
        device = x.device
        half_dim = self.t_dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class GaussianDiffusionBase(nn.Module):
    def __init__(self, t, model, betas):
        """
        Args:
            t (int): 时间步数
            model (nn.Module): 用于通过噪声图像 x_t 和时间步数 t 预测重构图像 x_0 的神经网络模型
            betas (torch.Tensor): 平滑系数
        """
        super(GaussianDiffusionBase, self).__init__()

        self.model = model
        self.t = t

        self.betas = betas  # .to(torch.float64)    # 使用 float64 提高准确率
        assert len(betas.shape) == 1, "beta must be 1-D tensor"
        assert (betas >= 0).all() and (betas <= 1).all(), "betas must be in (0, 1)"

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.], device=self.betas.device),
            self.alphas_cumprod[:-1]
        ])
        self.alphas_cumprod_next = torch.cat([
            self.alphas_cumprod[1:],
            torch.tensor([0.], device=self.betas.device)
        ])

        # 用于：已知 x_t 和 pred_noise 预测 x_0
        # :math:`\frac{1}{\bar\alpha_t}`
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        # :math:`\sqrt{\frac{1}{\bar\alpha_t}-1}` <=> \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}
        self.sqrt_recipm1_alphas_bar = torch.sqrt((1. - self.alphas_cumprod) / self.alphas_cumprod)

        # 前向传播 q(x_t | x_{t-1}) 过程参数
        # 用于：已知 x_0 和 pred_noise 计算 x_t
        # :math:`\sqrt{\bar\alpha_t}`
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # :math:`\sqrt{1-\bar\alpha_t}`
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def predict_x0(self, x_t, t, pred_noise):
        r"""
        已知 x_t 和 pred_noise 预测 x_0
        .. math::
            由 x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon_0
            得 x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_0}{\sqrt{\bar\alpha_t}}
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * pred_noise
        )

    def predict_xt(self, x_0, t, noise):
        r"""
        已知 x_0 和 pred_noise 预测 x_t
        .. math::
            x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon_0
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )


class GaussianDiffusionTrainer(GaussianDiffusionBase):
    def __init__(self, t, model, betas, loss_func):
        """
        Args:
            t (int): 时间步数
            model (nn.Module): 神经网络模型
            betas (torch.Tensor): 平滑系数
            loss_func (nn.Module): 损失函数
        """
        super(GaussianDiffusionTrainer, self).__init__(t, model, betas)
        self.loss_func = loss_func

    def forward(self, x_0, weights=1.0, sampling_type=SamplingType.STOCHASTIC):
        """
        Args:
            x_0 (torch.Tensor): 原始高斯噪声图像
            weights (float|torch.Tensor): 权重
            sampling_type (str): 采样类型，'stochastic' 或 'antithetic'
        """
        def q_sample(_noise):
            r"""
            从 q(x_t | x_0) 中采样噪声
            :math:`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon`
            """
            assert _noise.shape == x_0.shape, "noise shape must be the same as x_0"
            sample = self.predict_xt(x_0, t, _noise)
            return sample

        batch_size = x_0.shape[0]
        # 随机采样的方式
        if sampling_type == SamplingType.ANTITHETIC:
            t = torch.randint(0, self.t, size=(batch_size // 2 + 1,), device=x_0.device)
            t = torch.cat([t, self.t - t - 1], dim=0)[:batch_size]
        else:  # sampling_type == SamplingType.STOCHASTIC:
            t = torch.randint(0, self.t, size=(batch_size,), device=x_0.device)

        # 计算前向传播第 t 步的预测图像
        noise = torch.randn_like(x_0)
        x_t = q_sample(noise)            # 采样噪声得到 x_t
        pred_noise = self.model(x_t, t)  # 通过 x_t 和 t 预测噪声

        # 计算预测图像 x_recon 和原图像 x_0 的损失
        loss = self.loss_func(pred_noise, noise, weights)
        return loss


class DDPMSampler(GaussianDiffusionBase):
    """
    扩散模型 DDPM 的采样器
    论文链接：https://arxiv.org/abs/2006.11239
    """
    def __init__(self, t, model, betas, accelerate=True):
        super(DDPMSampler, self).__init__(t, model, betas)
        # 反向传播 q(x_{t-1} | x_t, x_0) 过程参数
        # :math:`\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`
        self.variance = (1 - self.alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sigma = torch.sqrt(self.variance)

        self.accelerate = accelerate
        if accelerate:
            # 已知 x_t 和 pred_noise 预测 x_t-1 的均值和标准差，用于计算预测噪声的均值 \miu
            # :math:`\sqrt{frac{1}{\alpha_t}}`
            self.posterior_mean_coeff_xt = torch.sqrt(1. / self.alphas)
            # :math:`frac{1-alpha_t}{\sqrt{\alpha_t}}{\sqrt{1-\bar\alpha_t}}`
            self.posterior_mean_coeff_eps = ((1. - self.alphas)
                                             / (torch.sqrt(1. - self.alphas_cumprod) * torch.sqrt(self.alphas)))
        else:
            # 通过 pred_noise 和 x_0 预测 x_t-1 的均值和标准差时，用于计算预测噪声的均值 \miu
            # :math:`\frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)}{1-\bar\alpha_t}`
            self.posterior_mean_coeff_x0 = (torch.sqrt(self.alphas_cumprod_prev) * (1. - self.alphas)
                                            / (1. - self.alphas_cumprod))
            # :math:`\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`
            self.posterior_mean_coeff_xt = ((1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
                                            / (1. - self.alphas_cumprod))

    def forward(self, x_t, clip_denoised=True):
        batch_size = x_t.shape[0]

        # x_t: torch.Tensor = torch.randn_like(x, device=x_t.device, requires_grad=False)  # 生成原始高斯噪声图像
        # x_t = torch.clamp(x_t * 0.5 + 0.5, 0, 1)
        for i in tqdm(list(reversed(range(0, self.t)))):            # 逐步加噪声
            t = x_t.new_ones((batch_size, ), dtype=torch.long) * i
            x_t = self.p_sample(x_t, t, i, clip_denoised)           # 反向采样

        x_0 = x_t
        return x_0.clip(-1, 1) * 0.5 + 0.5  # [0 ~ 1]

    def p_sample(self, x_t, t, time_step, clip_denoised=True):
        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值
        def predict_xt_prev_mean_from_noise(pred_noise):
            return (
                extract(self.posterior_mean_coeff_xt, time_step, x_t.shape) * x_t
                - extract(self.posterior_mean_coeff_eps, time_step, x_t.shape) * pred_noise
            )

        def q_mean_std(x_0):
            r"""
            计算后验概率分布 q(x_t | x_0) 的均值和标准差
            .. math::
                q(x_t | x_{t-1},x_0)
                \propto
                \mathcal{N}(
                    x_{t-1};
                    \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_t+\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{{1-\bar\alpha_t}},
                    \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathrm{I}
                )
            其中 posterior_mean 相当于分布中的均值\n
            posterior_variance 相当于分布中的方差\n
            Return: (mean, std)
            """
            # 后验均值
            posterior_mean = (
                extract(self.posterior_mean_coeff_x0, time_step, x_t.shape) * x_0
                + extract(self.posterior_mean_coeff_xt, time_step, x_t.shape) * x_t
            )
            # 后验方差
            posterior_std = extract(self.sigma, time_step, x_t.shape)
            return posterior_mean, posterior_std

        # 已知 x_t 和 pred_noise 得到 x_0，再通过 x_t 和 x_0 预测 x_t-1 的均值和标准差
        def p_mean_std():
            """
            通过将 x_t 和 t 传入模型预测噪声 p(x_{t-1} | x_t)，并根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            """
            pred_noise = self.model(x_t, t)                # 预测的噪声
            x_recon = self.predict_x0(x_t, t, pred_noise)  # 根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            if clip_denoised:
                x_recon.clamp_(-1, 1)                  # 使结果更加稳定

            xt_prev_mean, xt_prev_std = q_mean_std(x_recon)
            return xt_prev_mean, xt_prev_std

        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值和标准差
        def p_mean_std_accelerate():
            pred_noise = self.model(x_t, t)                             # 预测的噪声
            xt_prev_mean = predict_xt_prev_mean_from_noise(pred_noise)  # 根据原始噪声 (x_t) 和 pred_noise 预测 x_{t-1}

            xt_prev_std = extract(self.sigma, time_step, x_t.shape)
            return xt_prev_mean, xt_prev_std

        get_mean_std = p_mean_std_accelerate if self.accelerate else p_mean_std
        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 else 0.
        mean, std = get_mean_std()
        xt_prev = mean + std * noise
        return xt_prev


class DDIMSampler(GaussianDiffusionBase):
    """
    扩散模型 DDIM 的采样器
    论文链接：https://arxiv.org/abs/2010.02502
    相比于 DDPM，DDIM 的采用有 eta 介入，eta == 0 时，方差 == 0
    \n并且 DDIM 的扩散过程不依赖马尔可夫链，可以跳步
    """
    def __init__(self, t, timesteps: list[int], model, betas, eta=0, accelerate=True):
        """
        Args:
            t (int): 总的时间步数
            timesteps (list[int]): 时间步数列表，比如 t = 100 时可以取 timesteps = [10, 20, 30, 40, 50, 75, 100]
            model: 噪声生成模型
            betas: 插值
            eta: eta 取值范围为 [0, 1]，取值越接近 0，收敛所需步数越少，但是收敛值大（1000步内收敛效果最好）；
                 取值越接近 1，收敛所需步数越多，但是收敛值小（1000步后收敛效果最好）
            accelerate (bool): 是否启用加速（提前算好数据）
        """
        super(DDIMSampler, self).__init__(t, model, betas)
        self.eta = eta
        timesteps_curr = timesteps[1:]
        timesteps_prev = timesteps[:-1]
        self.timesteps = timesteps_curr

        self.alphas_cumprod = self.alphas_cumprod[timesteps_curr]
        self.alphas_cumprod_prev = self.alphas_cumprod_prev[timesteps_prev]
        self.alphas = self.alphas_cumprod / self.alphas_cumprod_prev

        # 用于计算预测噪声的方差 \sigma^2 = \delta_t^2 = variance
        self.variance = ((1. - self.alphas)
                         * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.sigma = eta * torch.sqrt(self.variance)
        sigma_square = (eta ** 2) * self.variance

        self.accelerate = accelerate
        if accelerate:
            # 通过 pred_noise 和 x_t 预测 x_t-1 的均值和标准差时，用于计算预测噪声的均值 \miu
            self.posterior_mean_coeff_xt = torch.sqrt(1. / self.alphas)
            self.posterior_mean_coeff_eps = (
                -torch.sqrt(1. - self.alphas_cumprod) * torch.sqrt(1. / self.alphas)
                + torch.sqrt(1 - self.alphas_cumprod_prev - sigma_square)
            )
        else:
            # 通过 pred_noise 和 x_0 预测 x_t-1 的均值和标准差时，用于计算预测噪声的均值 \miu
            self.posterior_mean_coeff_x0 = torch.sqrt(self.alphas_cumprod_prev)
            self.posterior_mean_coeff_eps = torch.sqrt(1 - self.alphas_cumprod_prev - sigma_square)

    def forward(self, x_t, clip_denoised=True):
        """
        Args:
            x_t (torch.Tensor): 原始高斯噪声图像
            clip_denoised (bool): 是否对重构图像进行裁剪
        """
        batch_size = x_t.shape[0]

        # x_t: torch.Tensor = torch.randn_like(x, device=x_t.device, requires_grad=False)  # 生成原始高斯噪声图像
        # x_t = torch.clamp(x_t * 0.5 + 0.5, 0, 1)
        for i in tqdm(list(reversed(range(0, len(self.timesteps))))):  # 逐步加噪声
            t = x_t.new_ones((batch_size, ), dtype=torch.long) * self.timesteps[i]
            x_t = self.p_sample(x_t, t, i, clip_denoised)    # 反向采样

        x_0 = x_t
        return x_0.clip(-1, 1) * 0.5 + 0.5  # [0 ~ 1]

    def p_sample(self, x_t, t, time_step, clip_denoised=True):
        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值
        def predict_xt_prev_mean_from_noise(pred_noise):
            return (
                extract(self.posterior_mean_coeff_xt, time_step, x_t.shape) * x_t
                + extract(self.posterior_mean_coeff_eps, time_step, x_t.shape) * pred_noise
            )

        def q_mean_std(x_0, pred_noise):
            r"""
            后验概率分布
            .. math::
                q(x_t | x_{t-1},x_0)
                \propto
                \mathcal{N}\left(
                    x_{t-1};
                    \sqrt{\bar{\alpha}_{t-1}}x_0
                        +\sqrt{1-\bar{\alpha}_{t-1}-\delta_{t}^{2}}\epsilon_{t},
                    \delta_{t}^{2} \mathcal{I}\right)
            其中 posterior_mean 相当于分布中的均值\n
            posterior_variance 相当于分布中的方差\n
            posterior_log_variance 是对 posterior_variance 取对数，结果会更加稳定
            """
            # 后验均值
            posterior_mean = (
                extract(self.posterior_mean_coeff_x0, time_step, x_t.shape) * x_0
                + extract(self.posterior_mean_coeff_eps, time_step, x_t.shape) * pred_noise  # direction pointing to x_t
            )
            # 后验标准差
            posterior_std = extract(self.sigma, time_step, x_t.shape)
            return posterior_mean, posterior_std

        # 已知 x_t 和 pred_noise 得到 x_0，再通过 pred_noise 和 x_0 预测 x_t-1 的均值和标准差
        def p_mean_std():
            pred_noise = self.model(x_t, t)                 # 预测的噪声
            x_recon = self.predict_x0(x_t, t, pred_noise)   # 根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            if clip_denoised:
                x_recon.clamp_(-1, 1)                  # 使结果更加稳定

            xt_prev_mean, xt_prev_std = q_mean_std(x_recon, pred_noise)
            return xt_prev_mean, xt_prev_std

        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值和标准差
        def p_mean_std_accelerate():
            pred_noise = self.model(x_t, t)                              # 预测的噪声
            xt_prev_mean = predict_xt_prev_mean_from_noise(pred_noise)   # 根据原始噪声 (x_t) 和 pred_noise 预测 x_{t-1}

            xt_prev_std = extract(self.sigma, time_step, x_t.shape)
            return xt_prev_mean, xt_prev_std

        get_mean_std = p_mean_std_accelerate if self.accelerate else p_mean_std
        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 and self.eta != 0 else torch.zeros_like(x_t)
        mean, std = get_mean_std()
        xt_prev = mean + std * noise

        return xt_prev

    @staticmethod
    def space_timesteps(t, section_strides, section_counts=None):
        """
        如果 section_counts == None，则将 num_timesteps 分为 len(section_counts) 个部分，每部分包含 section_counts[i] 的时间步长
        否则，将 num_timesteps 分为 len(section_counts) 个部分，每部分包含 section_counts[i] 个 section_counts[i] 的时间步长
        Args:
            t (int): 总的时间步数
            section_strides (list): 每部分的时间步数，可以取[10, 15, 20, 25]
            section_counts (list): 每部分的数量，可以取[10, 6, 5, 4]
        """
        def check_overflow():
            return t_current >= t

        def append_and_check_overflow():
            if check_overflow():
                return True
            else:
                all_timesteps.append(t_current)

        size_per = t // len(section_strides)
        t_current = 1
        all_timesteps = [0, t_current]
        if section_counts is None:
            # 将 num_timesteps 分为 len(section_counts) 个部分，每部分包含 section_counts[i] 的时间步长
            while True:
                t_current += section_strides[t_current // size_per]
                if append_and_check_overflow():
                    break
        else:
            # 将 num_timesteps 分为 len(section_counts) 个部分，每部分包含 section_counts[i] 个 section_counts[i] 的时间步长，
            for stride, count in zip(section_strides, section_counts):
                assert stride > 0, "section_strides must be positive"
                for i in range(count):
                    t_current += stride
                    if append_and_check_overflow():
                        break
                if check_overflow():
                    break
            # 如果剩余的时间步数大于 section_counts[-1]，则将剩下的部分使用 section_counts[-1] 填充
            while t_current < t:
                t_current += section_strides[-1]
                if append_and_check_overflow():
                    break
        return all_timesteps


class Diffusion(nn.Module):
    def __init__(self, t, model, betas, loss_func):
        super(Diffusion, self).__init__()

        self.trainer = GaussianDiffusionTrainer(t, model, betas, loss_func)
        self.sampler = DDPMSampler(t, model, betas)

    def forward(self, x):
        return self.sample(x)

    def sample(self, x):
        return self.sampler(x)

    def loss(self, x, weights=1.0):
        return self.trainer(x, weights)


if __name__ == '__main__':
    _t = 1000
    _all_timesteps = DDIMSampler.space_timesteps(_t, [10, 15, 20, 25])
    sampler = DDIMSampler(_t, _all_timesteps, None, linear(1e-4, 0.02, 1000), eta=0, accelerate=True)
    print(_all_timesteps)

