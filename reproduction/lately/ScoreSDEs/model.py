import math
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import integrate

from utils.torch import expend_as
from modules.diffusion.sampler import MarkovChainNoiseSampler


def marginal_prob_std(t, sigma, device):
    """
    计算 :math:`p_{0t}(x(t) | x(0))` 的标准差
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / math.log(sigma))


class ScoreSDETrainer(nn.Module):
    sigma = 25

    def __init__(self, model):
        """
        Args:
            model (nn.Module): 噪声生成神经网络模型
        """
        super(ScoreSDETrainer, self).__init__()
        self.model = model

    def forward(self, x_0, eps=1e-5):
        """
        Args:
            x_0 (torch.Tensor): 原始高斯噪声图像
            eps (float): 随机噪声的最小值，保证数值稳定性
        """
        def predict_xt(_noise):
            r"""
            已知 x_0 和 pred_noise 预测 x_t
            .. math::
                x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon_0
            """
            _std = marginal_prob_std(t, self.sigma, device=x_0.device)
            return x_0 + _noise * expend_as(_std, x_0), _std

        def q_sample(_noise):
            r"""
            从 q(x_t | x_0) 中采样噪声
            :math:`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon`
            """
            assert _noise.shape == x_0.shape, "noise shape must be the same as x_0"
            sample, _std = predict_xt(_noise)
            return sample, _std

        batch_size = x_0.shape[0]
        # 随机采样
        t = torch.rand(batch_size, device=x_0.device) * (1 - eps) + eps

        # 计算前向传播第 t 步的预测图像
        noise = torch.randn_like(x_0)
        x_t, std = q_sample(noise)            # 采样噪声得到 x_t
        pred_noise = self.model(x_t, t)  # 通过 x_t 和 t 预测噪声

        # 计算预测图像 x_recon 和原图像 x_0 的损失
        loss = torch.mean(torch.sum((pred_noise * expend_as(std, x_0) + noise) ** 2, dim=(1, 2, 3)))  # self.loss_func(pred_noise, noise, weights)
        return loss

    def loss_func(self, x_0, eps=1e-5):
        return self.forward(x_0, eps)


class EulerMaruyamaSampler(MarkovChainNoiseSampler):
    sigma = 25

    def __init__(self, model, t, betas):
        super().__init__(model, t)
        self.betas = betas[:-1]                  # [0, ..., 1]
        self.step_size = betas[1:] - betas[:-1]  # [-, ..., 1] - [0, ..., -]

    def sample(self, x_t, t, time_step, **kwargs):
        def p_mean_std():
            """
            通过将 x_t 和 t 传入模型预测噪声 p(x_{t-1} | x_t)，并根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            """
            pred_noise = self.model(x_t, t)  # 预测的噪声 s_\theta(x_t, t)

            sigma = torch.tensor(self.sigma ** t, device=x_t.device)
            xt_prev_mean = x_t + expend_as(sigma ** 2, x_t) * pred_noise * self.step_size[t]
            xt_prev_std = expend_as(sigma, x_t) * torch.sqrt(self.step_size[t])
            return xt_prev_mean, xt_prev_std

        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 else 0.
        mean, std = p_mean_std()
        xt_prev = mean + std * noise
        return xt_prev


class PredictorCorrectorSampler(MarkovChainNoiseSampler):
    # snr 即 signal-to-noise ratio，即信噪比
    snr = 0.16

    def __init__(self, model, t, betas, sampler):
        super().__init__(model, t)
        self.betas = betas[:-1]                  # [0, ..., 1]
        self.step_size = betas[1:] - betas[:-1]  # [-, ..., 1] - [0, ..., -]

        self.sampler = sampler

    def sample(self, x_t, t, time_step, **kwargs):
        def corrector_mean_std():
            """
            通过将 x_t 和 t 传入模型预测噪声 p(x_{t-1} | x_t)，并根据原始噪声 (x_t) 和 pred_noise 预测 x_0 (重构图像)
            """
            pred_grad = self.model(x_t, t)  # 预测的噪声 s_\theta(x_t, t)
            pred_grad_norm = torch.norm(pred_grad.view(pred_grad.shape[0], -1), dim=-1).mean()
            pred_noise_norm = torch.sqrt(torch.cumprod(x_t.shape[1:], dim=0))
            langevin_step_size = 2 * (self.snr * pred_noise_norm / pred_grad_norm) ** 2

            xt_prev_mean = x_t + langevin_step_size * pred_grad
            xt_prev_std = torch.sqrt(2 * langevin_step_size)
            return xt_prev_mean, xt_prev_std

        # Corrector step: 郎之万动力学马尔科夫链 (Langevin MCMC)
        # t != 0 时，添加噪声；当 t == 0 时，不添加噪声
        noise = torch.randn_like(x_t) if time_step > 0 else 0.
        mean, std = corrector_mean_std()
        xt_prev = mean + std * noise

        # Predictor step: 使用 Euler-Maruyama 进行噪声预测
        xt_prev = self.sampler.p_sample(xt_prev, t, time_step, **kwargs)
        return xt_prev


class NumericalODESampler(nn.Module):
    # 黑盒 ODE 解算器的容许误差
    error_tolerance = 1e-5

    def __init__(self, model, t, beta_start=1e-5, beta_end=1,
                 absolute_tolerance=error_tolerance, relative_tolerance=error_tolerance):
        super().__init__()
        self.t = t
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

        self.model = model

    # noinspection PyUnresolvedReferences
    def forward(self, x_t, clip_denoised=True):
        batch_size = x_t.shape[0]

        t = torch.ones(batch_size, device=x_t.device)  # 时间步 t

        def ode_solver():
            pred_noise = self.model(x_t, t)  # 预测的噪声 s_\theta(x_t, t)

            sigma = torch.tensor(self.sigma ** t, device=x_t.device)
            return -0.5 * (sigma ** 2) * pred_noise

        # 黑盒 ODE 解算器
        result = integrate.solve_ivp(ode_solver,
                                     (self.beta_end, self.beta_start),
                                     x_t.reshape(-1).cpu().numpy(),
                                     rtol=self.relative_tolerance,
                                     atol=self.absolute_tolerance,
                                     method='RK45')
        x_0 = torch.tensor(result.y[:, -1], device=x_t.device).reshape(x_t.shape)
        return x_0

