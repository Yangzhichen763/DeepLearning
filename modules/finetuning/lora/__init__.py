import torch
import torch.nn as nn


class LoRA(nn.Module):
    """
    LoRA: Low-Rank Adaptation
    论文链接 2021：https://arxiv.org/abs/2106.09685
    通常将 LoRA 更新到线性层的输出中，即将 linear(x) 改为 linear(x) + lora(x)
    """
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoRA, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.w_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.w_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.w_a @ self.w_b)
        return x


class LinearWithLoRA(nn.Module):
    """
    线性层 + LoRA 层
    """
    def __init__(self, linear, rank, alpha):
        super(LinearWithLoRA, self).__init__()
        self.linear = linear
        self.lora = LoRA(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        x = self.linear(x) + self.lora(x)
        return x


