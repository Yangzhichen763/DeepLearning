import torch
import torch.nn as nn
from tqdm import tqdm

from abc import ABCMeta, abstractmethod


class MarkovChainNoiseSampler(nn.Module):
    """
    Base class for noise sampler.
    噪声采样器的基类。
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, t):
        super().__init__()
        self.model = model
        self.t = t

    def forward(self, x_t, clip_denoised=True):
        batch_size = x_t.shape[0]

        # x_t: torch.Tensor = torch.randn_like(x, device=x_t.device, requires_grad=False)  # 生成原始高斯噪声图像
        # x_t = torch.clamp(x_t * 0.5 + 0.5, 0, 1)
        for i in tqdm(list(reversed(range(0, self.t)))):               # 逐步加噪声
            t = x_t.new_ones((batch_size, ), dtype=torch.long) * i
            x_t = self.sample(x_t, t, i, clip_denoised=clip_denoised)  # 反向采样

        x_0 = x_t
        return x_0.clip(-1, 1) * 0.5 + 0.5  # [0 ~ 1]

    @abstractmethod
    def sample(self, x_t, t, step, **kwargs):
        pass
