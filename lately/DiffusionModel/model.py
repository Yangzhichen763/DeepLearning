import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lossFunc.WeightedLoss import *
from interpret import *
from modules import UNet


def extract(a, t, x_shape):
    """
    将 a 按照 t 进行切片，并将切片后的结果展平为 shape=[batch_size, 1, 1, 1, ...]
    Args:
        a (torch.Tensor):
        t (torch.LongTensor):
        x_shape (tuple|torch.Size):
    """
    out = a.gather(dim=-1, index=t)
    return out.view(-1, *((1,) * (len(x_shape) - 1)))


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


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, t, model, betas, loss_func):
        super(GaussianDiffusionTrainer, self).__init__()

        self.model = model
        self.t = t
        self.loss_func = loss_func

        self.register_buffer("betas", betas)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # 前向传播 q(x_t | x_{t-1}) 过程参数
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, weights=1.0):
        def q_sample(_t, _noise):
            r"""
            :math:`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon`
            """
            sample = (
                extract(self.sqrt_alphas_bar, _t, x_0.shape) * x_0
                + extract(self.sqrt_one_minus_alphas_bar, _t, x_0.shape) * _noise
            )
            return sample

        batch_size = x_0.shape[0]
        t = torch.randint(0, self.t, size=(batch_size,), device=x_0.device)

        # 计算前向传播第 t 步的预测图像
        noise = torch.randn_like(x_0)
        x_t = q_sample(t, noise)        # 采样噪声得到 x_t
        x_recon = self.model(x_t, t)    # 通过 x_t 和 t 预测 x_0

        # 计算预测图像 x_recon 和原图像 x_0 的损失
        loss = self.loss_func(x_recon, noise, weights)
        return loss


class DiffusionModelSamplerBase(nn.Module):
    def __init__(self):
        super(DiffusionModelSamplerBase, self).__init__()

    def init(self, alphas_bar: torch.Tensor):
        # 用于：已知 x_t 和 pred_noise 预测 x_0
        # :math:`\frac{1}{\bar\alpha_t}`
        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1. / alphas_bar))
        # :math:`\sqrt{\frac{1}{\bar\alpha_t}-1}` <=> \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}
        self.register_buffer("sqrt_recip_alphas_bar_minus_one", torch.sqrt(1. / alphas_bar - 1))

    def predict_x0_from_noise(self, x_t, t, pred_noise):
        r"""
        已知 x_t 和 pred_noise 预测 x_0
        .. math::
            由 x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon_0
            得 x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_0}{\sqrt{\bar\alpha_t}}
        """
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
                - extract(self.sqrt_recip_alphas_bar_minus_one, t, x_t.shape) * pred_noise
        )


class DDPMSampler(DiffusionModelSamplerBase):
    def __init__(self, t, model, betas, method=True):
        super(DDPMSampler, self).__init__()

        self.model = model
        self.t = t

        self.register_buffer("betas", betas)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, (1, 0), value=1.)[:t]
        super(DDPMSampler, self).init(alphas_bar)

        # 反向传播过程参数
        # :math:`\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`
        self.register_buffer(
            "variance",
            (1 - alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)
        )
        self.register_buffer(
            "sigma",
            torch.sqrt(self.variance)
        )

        self.method = method
        if method:
            # 通过 pred_noise 和 x_0 预测 x_t-1 的均值和标准差时，用于计算预测噪声的均值 \miu
            # :math:`\frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)}{1-\bar\alpha_t}`
            self.register_buffer(
                "mean_coeff_x0",
                torch.sqrt(alphas_bar_prev) * (1. - alphas) / (1. - alphas_bar)
            )
            # :math:`\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`
            self.register_buffer(
                "mean_coeff_xt",
                (1. - alphas_bar_prev) * torch.sqrt(alphas) / (1. - alphas_bar)
            )
        else:
            # 已知 x_t 和 pred_noise 预测 x_t-1 的均值和标准差，用于计算预测噪声的均值 \miu
            # :math:`\sqrt{frac{1}{\alpha_t}}`
            self.register_buffer(
                "mean_coeff_xt",
                torch.sqrt(1. / alphas)
            )
            # :math:`frac{1-alpha_t}{\sqrt{\alpha_t}}{\sqrt{1-\bar\alpha_t}}`
            self.register_buffer(
                "mean_coeff_eps",
                self.mean_coeff_xt * (1. - alphas) / torch.sqrt(1. - alphas_bar)
            )

    def forward(self, x_t):
        batch_size = x_t.shape[0]

        # x_t: torch.Tensor = torch.randn_like(x, device=x_t.device, requires_grad=False)  # 生成原始高斯噪声图像
        # x_t = torch.clamp(x_t * 0.5 + 0.5, 0, 1)
        for i in tqdm(reversed(range(0, self.t))):                  # 逐步加噪声
            t = x_t.new_ones((batch_size, ), dtype=torch.long) * i
            x_t = self.p_sample(x_t, t, i)                          # 反向采样

        x_0 = x_t
        return x_0.clip(-1, 1)  # * 0.5 + 0.5  # [0 ~ 1]

    def p_sample(self, x_t, t, time_step):
        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值
        def predict_xt_prev_mean_from_noise(pred_noise):
            return (
                extract(self.mean_coeff_xt, t, x_t.shape) * x_t
                - extract(self.mean_coeff_eps, t, x_t.shape) * pred_noise
            )

        def q_mean_std(x_0):
            r"""
            后验概率分布
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
            """
            # 后验均值
            posterior_mean = (
                extract(self.mean_coeff_x0, t, x_t.shape) * x_0
                + extract(self.mean_coeff_xt, t, x_t.shape) * x_t
            )
            # 后验方差
            posterior_std = extract(self.sigma, t, x_t.shape)
            return posterior_mean, posterior_std

        # 已知 x_t 和 pred_noise 得到 x_0，再通过 x_t 和 x_0 预测 x_t-1 的均值和标准差
        def p_mean_std_1():
            pred_noise = self.model(x_t, t)                             # 预测的噪声
            x_recon = self.predict_x0_from_noise(x_t, t, pred_noise)    # 根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            x_recon.clamp_(-1, 1)                                  # 使结果更加稳定

            xt_prev_mean, xt_prev_std = q_mean_std(x_recon)
            return xt_prev_mean, xt_prev_std

        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值和标准差
        def p_mean_std_2():
            pred_noise = self.model(x_t, t)                              # 预测的噪声
            xt_prev_mean = predict_xt_prev_mean_from_noise(pred_noise)   # 根据原始噪声 (x_t) 和 pred_noise 预测 x_{t-1}

            xt_prev_std = extract(self.sigma, t, x_t.shape)
            return xt_prev_mean, xt_prev_std

        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 else 0.
        mean, std = p_mean_std_1() if self.method else p_mean_std_2()
        xt_prev = mean + std * noise
        return xt_prev


class DDIMSampler(DiffusionModelSamplerBase):
    def __init__(self, t, model, betas, miu=0, method=True):
        super(DDIMSampler, self).__init__()

        self.model = model
        self.t = t
        self.miu = miu

        self.register_buffer("betas", betas)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, (1, 0), value=1.)[:t]
        super(DDIMSampler, self).init(alphas_bar)

        # 用于计算预测噪声的方差 \sigma^2 = \delta_t^2 = variance
        self.register_buffer(
            "variance",
            (1 - alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            "sigma",
            miu * torch.sqrt(self.variance))

        self.method = method
        if method:
            # 通过 pred_noise 和 x_0 预测 x_t-1 的均值和标准差时，用于计算预测噪声的均值 \miu
            self.register_buffer(
                "mean_coeff_x0",
                torch.sqrt(alphas_bar_prev)
            )
            self.register_buffer(
                "mean_coeff_eps",
                torch.sqrt(1 - alphas_bar_prev - self.sigma ** 2)
            )
        else:
            # 通过 x_t 和 x_0 预测 x_t-1 的均值和标准差时，用于计算预测噪声的均值 \miu
            self.register_buffer(
                "mean_coeff_xt",
                torch.sqrt(1. / alphas)
            )
            self.register_buffer(
                "mean_coeff_eps",
                self.mean_coeff_xt * torch.sqrt(1. - alphas_bar) - torch.sqrt(1 - alphas_bar_prev - self.sigma ** 2)
            )

    def forward(self, x_t):
        batch_size = x_t.shape[0]

        # x_t: torch.Tensor = torch.randn_like(x, device=x_t.device, requires_grad=False)  # 生成原始高斯噪声图像
        # x_t = torch.clamp(x_t * 0.5 + 0.5, 0, 1)
        for i in tqdm(reversed(range(0, self.t))):                  # 逐步加噪声
            t = x_t.new_ones((batch_size, ), dtype=torch.long) * i
            x_t = self.p_sample(x_t, t, i)                          # 反向采样

        x_0 = x_t
        return x_0.clip(-1, 1)  # * 0.5 + 0.5  # [0 ~ 1]

    def p_sample(self, x_t, t, time_step):
        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值
        def predict_xt_prev_mean_from_noise(pred_noise):
            return (
                extract(self.mean_coeff_xt, t, x_t.shape) * x_t
                - extract(self.mean_coeff_eps, t, x_t.shape) * pred_noise
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
                extract(self.mean_coeff_x0, t, x_t.shape) * x_0
                + extract(self.mean_coeff_eps, t, x_t.shape) * pred_noise   # direction pointing to x_t
            )
            # 后验标准差
            posterior_std = extract(self.sigma, t, x_t.shape)
            return posterior_mean, posterior_std

        # 已知 x_t 和 pred_noise 得到 x_0，再通过 pred_noise 和 x_0 预测 x_t-1 的均值和标准差
        def p_mean_std_1():
            pred_noise = self.model(x_t, t)                             # 预测的噪声
            x_recon = self.predict_x0_from_noise(x_t, t, pred_noise)    # 根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)

            xt_prev_mean, xt_prev_std = q_mean_std(x_recon, pred_noise)
            return xt_prev_mean, xt_prev_std

        # 已知 x_t 和 pred_noise 预测 x_t-1 的均值和标准差
        def p_mean_std_2():
            pred_noise = self.model(x_t, t)                              # 预测的噪声
            xt_prev_mean = predict_xt_prev_mean_from_noise(pred_noise)   # 根据原始噪声 (x_t) 和 pred_noise 预测 x_{t-1}

            xt_prev_std = extract(self.sigma, t, x_t.shape)
            return xt_prev_mean, xt_prev_std

        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 and self.miu != 0 else 0.
        mean, std = p_mean_std_1() if self.method else p_mean_std_2()
        xt_prev = mean + std * noise
        return xt_prev


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
    _batch_size = 3
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _pred = torch.randn(_batch_size, 3, 64, 64).to(_device)
    _target = torch.randn(_batch_size, 3, 64, 64).to(_device)
    _t = 100
    # _model = Diffusion(t=_t,
    #                    model=UNet(t=_t, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
    #                               num_res_blocks=2, dropout=0.15).to(_device),
    #                    betas=linear_interpret(1e-4, 2e-2, 100, device=_device),
    #                    loss_func=WeightedL2Loss())
    # _loss = _model.loss(_target)
    # print(f"loss: {_loss.mean().item()}")
    _sampler = DDPMSampler(t=_t,
                           model=UNet(t=_t, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                                      num_res_blocks=2, dropout=0.15).to(_device),
                           betas=linear_interpret(1e-4, 2e-2, 100, device=_device))
    _sampler(_pred)

