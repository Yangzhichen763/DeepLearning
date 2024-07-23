import torch
import torch.nn as nn
from tqdm import tqdm

from lossFunc.WeightedLoss import *
from ..interpret import *
from utils.torch import extract

import enum


class SamplingType(enum.Enum):
    STOCHASTIC = enum.auto()
    ANTITHETIC = enum.auto()


class DDPMBase(nn.Module):
    def __init__(self, t, model, betas):
        """
        Args:
            t (int): 时间步数
            model (nn.Module): 用于通过噪声图像 x_t 和时间步数 t 预测重构图像 x_0 的神经网络模型
            betas (torch.Tensor): 平滑系数
        """
        super(DDPMBase, self).__init__()

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
            extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t) * pred_noise
        )

    def predict_xt(self, x_0, t, noise):
        r"""
        已知 x_0 和 pred_noise 预测 x_t
        .. math::
            x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon_0
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0) * noise
        )


class DDPMTrainer(DDPMBase):
    def __init__(self, t, model, betas, loss_func):
        """
        Args:
            t (int): 时间步数
            model (nn.Module): 神经网络模型
            betas (torch.Tensor): 平滑系数
            loss_func (nn.Module): 损失函数
        """
        super(DDPMTrainer, self).__init__(t, model, betas)
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
        elif sampling_type == SamplingType.STOCHASTIC:
            t = torch.randint(0, self.t, size=(batch_size,), device=x_0.device)
        else:
            raise NotImplementedError(f"Not implemented sampling_type: {sampling_type}")

        # 计算前向传播第 t 步的预测图像
        noise = torch.randn_like(x_0)
        x_t = q_sample(noise)            # 采样噪声得到 x_t
        pred_noise = self.model(x_t, t)  # 通过 x_t 和 t 预测噪声

        # 计算预测图像 x_recon 和原图像 x_0 的损失
        loss = self.loss_func(pred_noise, noise, weights)
        return loss


class DDPMSampler(DDPMBase):
    """
    扩散模型 DDPM 的采样器
    论文链接：https://arxiv.org/abs/2006.11239
    """
    def __init__(self, t, model, betas):
        super(DDPMSampler, self).__init__(t, model, betas)
        # 反向传播 q(x_{t-1} | x_t, x_0) 过程参数
        # :math:`\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`
        self.variance = (1 - self.alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sigma = torch.sqrt(self.variance)

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
                extract(self.posterior_mean_coeff_x0, time_step, x_t) * x_0
                + extract(self.posterior_mean_coeff_xt, time_step, x_t) * x_t
            )
            # 后验方差
            posterior_std = extract(self.sigma, time_step, x_t)
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

        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 else 0.
        mean, std = p_mean_std()
        xt_prev = mean + std * noise
        return xt_prev


class DDPM(nn.Module):
    def __init__(self, t, model, betas, loss_func):
        super(DDPM, self).__init__()

        self.trainer = DDPMTrainer(t, model, betas, loss_func)
        self.sampler = DDPMSampler(t, model, betas)

    def forward(self, x):
        return self.sample(x)

    def sample(self, x):
        return self.sampler(x)

    def loss(self, x, weights=1.0):
        return self.trainer(x, weights)


if __name__ == '__main__':
    _t = 1000
    _all_timesteps = DDPMSampler.space_timesteps(_t, [10, 15, 20, 25])
    sampler = DDPMSampler(_t, _all_timesteps, None, linear(1e-4, 0.02, 1000), eta=0, accelerate=True)
    print(_all_timesteps)

