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
        t = torch.randint(0, self.t, size=[batch_size], device=x_0.device, dtype=torch.long)

        # 计算前向传播第 t 步的预测图像
        noise = torch.randn_like(x_0)
        x_t = q_sample(t, noise)        # 采样噪声得到 x_t
        x_recon = self.model(x_t, t)    # 通过 x_t 和 t 预测 x_0

        # 计算预测图像 x_recon 和原图像 x_0 的损失
        loss = self.loss_func(x_recon, x_0, weights)
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, t, model, betas):
        super(GaussianDiffusionSampler, self).__init__()

        self.model = model
        self.t = t

        self.register_buffer("betas", betas)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, (1, 0), value=1.)[:t]

        # 反向传播过程参数
        # :math:`\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`
        posterior_variance = (1 - alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)
        self.register_buffer("variance", posterior_variance)
        self.register_buffer("log_variance_clipped", torch.log(posterior_variance).clamp(min=1e-20))

        # 用于：已知 x_t 和 pred_noise 预测 x_0
        # :math:`\frac{1}{\bar\alpha_t}`
        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1. / alphas_bar))
        # :math:`\sqrt{\frac{1}{\bar\alpha_t}-1}` <=> \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}
        self.register_buffer("sqrt_recip_alphas_bar_minus_one", torch.sqrt(1. / alphas_bar - 1))

        # 用于计算预测噪声的均值 \miu
        # :math:`\frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)}{1-\bar\alpha_t}`
        self.register_buffer(
            "mean_coeff1",
            torch.sqrt(alphas_bar_prev) * (1. - alphas) / (1. - alphas_bar)
        )
        # :math:`\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}`
        self.register_buffer(
            "mean_coeff2",
            (1. - alphas_bar_prev) * torch.sqrt(alphas) / (1. - alphas_bar)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        x_t: torch.Tensor = torch.randn_like(x, device=device, requires_grad=False)  # 生成原始高斯噪声图像
        for i in tqdm(reversed(range(0, self.t)), position=0):  # 逐步加噪声
            t = x_t.new_ones((batch_size, ), dtype=torch.long) * i
            x_t = self.p_sample(x_t, t, i)  # 反向采样

        x_0 = x_t
        return x_0.clip(-1, 1)

    def p_sample(self, x_t, t, time_step):
        """
        从噪声 y_t 加噪声得到 y_{t-1}
        以下函数的 x_0, x_t 与没有特定先后验的指代
        """

        # 已知 x_t 和 pred_noise 预测 x_0
        def predict_x0_from_noise(pred_noise):
            r"""
            已知 x_t 和 pred_noise，预测 x_0
            .. math::
                由 x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon_0
                得 x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_0}{\sqrt{\bar\alpha_t}}
            """
            return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
                - extract(self.sqrt_recip_alphas_bar_minus_one, t, x_t.shape) * pred_noise
            )

        # 求后验分布的均值和方差
        def q_mean_variance(x_0):
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
            posterior_log_variance 是对 posterior_variance 取对数，结果会更加稳定
            """
            # 后验均值
            posterior_mean = (
                extract(self.mean_coeff1, t, x_t.shape) * x_0
                + extract(self.mean_coeff2, t, x_t.shape) * x_t
            )
            # 后验方差，取对数可以使结果更加稳定
            posterior_log_variance = extract(self.log_variance_clipped, t, x_t.shape)
            return posterior_mean, posterior_log_variance

        # 求先验分布的均值和方差
        def p_mean_variance():
            pred_noise = self.model(x_t, t)                     # 预测的噪声
            x_recon = predict_x0_from_noise(pred_noise)         # 根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            x_recon.clamp_(-1, 1)                          # 使结果更加稳定

            xt_prev_mean, xt_prev_log_variance = q_mean_variance(x_recon)
            return xt_prev_mean, xt_prev_log_variance

        noise = torch.randn_like(x_t) if time_step > 0 else 0.
        model_mean, model_log_variance = p_mean_variance()      # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        model_variance = (0.5 * model_log_variance).exp()       # exp(0.5 * log(x)) = sqrt(x) 更加稳定
        xt_prev = model_mean + model_variance * noise
        return xt_prev


class Diffusion(nn.Module):
    def __init__(self, t, model, betas, loss_func):
        super(Diffusion, self).__init__()

        self.trainer = GaussianDiffusionTrainer(t, model, betas, loss_func)
        self.sampler = GaussianDiffusionSampler(t, model, betas)

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
    _model = Diffusion(t=100,
                       model=UNet(3, 3, 100, device=_device).to(_device),
                       betas=linear_interpret(1e-4, 2e-2, 100, device=_device),
                       loss_func=WeightedL2Loss())
    _out = _model.sample(_pred)
    _loss = _model.loss(_target)

    print(f"action: {_out}; loss: {_loss.mean().item()}")
