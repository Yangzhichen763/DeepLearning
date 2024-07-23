
import torch
import torch.nn as nn
from tqdm import tqdm

from ..DDPM import DDPMBase, DDPMTrainer
from utils.torch import extract


class DDIMSampler(DDPMBase):
    """
    扩散模型 DDIM 的采样器
    论文链接：https://arxiv.org/abs/2010.02502
    相比于 DDPM，DDIM 的采用有 eta 介入，eta == 0 时，方差 == 0
    \n并且 DDIM 的扩散过程不依赖马尔可夫链，可以跳步
    """
    def __init__(self, t, timesteps: list[int], model, betas, eta=0):
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
                extract(self.posterior_mean_coeff_x0, time_step, x_t) * x_0
                + extract(self.posterior_mean_coeff_eps, time_step, x_t) * pred_noise  # direction pointing to x_t
            )
            # 后验标准差
            posterior_std = extract(self.sigma, time_step, x_t)
            return posterior_mean, posterior_std

        # 已知 x_t 和 pred_noise 得到 x_0，再通过 pred_noise 和 x_0 预测 x_t-1 的均值和标准差
        def p_mean_std():
            pred_noise = self.model(x_t, t)                 # 预测的噪声
            x_recon = self.predict_x0(x_t, t, pred_noise)   # 根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            if clip_denoised:
                x_recon.clamp_(-1, 1)                  # 使结果更加稳定

            xt_prev_mean, xt_prev_std = q_mean_std(x_recon, pred_noise)
            return xt_prev_mean, xt_prev_std

        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 and self.eta != 0 else torch.zeros_like(x_t)
        mean, std = p_mean_std()
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


class DDIM(nn.Module):
    def __init__(self, t, model, betas, loss_func):
        super(DDIM, self).__init__()

        self.trainer = DDPMTrainer(t, model, betas, loss_func)
        self.sampler = DDIMSampler(t, model, betas)

    def forward(self, x):
        return self.sample(x)

    def sample(self, x):
        return self.sampler(x)

    def loss(self, x, weights=1.0):
        return self.trainer(x, weights)