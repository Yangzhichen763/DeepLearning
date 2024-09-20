import torch.nn as nn


"""
MetaFormer pooling module.
论文链接：https://arxiv.org/abs/2111.11418
"""


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x