import math

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

import sys
sys.path.append('..')
# noinspection PyUnresolvedReferences
from Attention import MultiheadAttention
sys.path.pop()


"""
PVT: Pyramid Vision Transformer
  - 传统 CNN backbone 的感受野随着深度逐渐增大，而 PVT 始终保持全局感受野，对检测、分割任务更为合适
  - 对于 ViT，在密集的像素级预测任务（如目标检测、分割等）上并不适合使用：
      - 输出分辨率比较低，且只有一个单一尺度，输出步幅为 32 或 16
      - 输入尺寸增大一点，造成计算复杂度和内存消耗的大幅增加

SRA: Spatial-Reduction Attention
论文链接 2021：https://arxiv.org/abs/2102.12122

Linear SRA: Linear Spatial-Reduction Attention
论文链接 2021：https://arxiv.org/abs/2106.13797
"""


class SRA(nn.Module):
    """
    Spatial-Reduction Attention (SRA) module.
    参考代码：https://github.com/whai362/PVT/blob/v2/classification/pvt.py
    """
    def __init__(self, num_heads, d_model, sr_ratio=1, dropout=0.1):
        super(SRA, self).__init__()

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(d_model)

        self.multihead_attention = MultiheadAttention(num_heads, d_model, dropout=dropout)

    # noinspection PyPep8Naming
    def forward(self, q, k, v, H, W, mask=None):
        """
        Args:
            q: query （要查询的信息）[B, q_n, dim]
            k: key   （被查询的向量）[B, kv_n, dim]
            v: value （查询得到的值）[B, kv_n, dim]
            H: 输入图像 kv 高
            W: 输入图像 kv 宽
            mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(q_n, kv_n, dtype=torch.bool))

        Returns:
        """
        B, N, C = q.shape

        # 获得 Q, K, V 矩阵
        def spatial_reduce(x: torch.Tensor):
            x = x.permute(0, 2, 1).view(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            return x
        k = spatial_reduce(k)
        v = spatial_reduce(v)

        q, attention = self.multihead_attention(q, k, v, mask=mask)
        return q, attention


class LinearSRA(nn.Module):
    """
    Linear Spatial-Reduction Attention (Linear SRA) module.
    参考代码：https://github.com/whai362/PVT/blob/v2/classification/pvt_v2.pyhttps://github.com/whai362/PVT/blob/v2/classification/pvt.py
    """
    def __init__(self, num_heads, d_model, sr_ratio=1, dropout=0.1):
        super(LinearSRA, self).__init__()

        self.sr_ratio = sr_ratio
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

        self.multihead_attention = MultiheadAttention(num_heads, d_model, dropout=dropout)

        self.apply(self._init_weights)

    # noinspection PyMethodMayBeStatic
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # noinspection PyPep8Naming
    def forward(self, q, k, v, H, W, mask=None):
        """
        Args:
            q: query （要查询的信息）[B, q_n, dim]
            k: key   （被查询的向量）[B, kv_n, dim]
            v: value （查询得到的值）[B, kv_n, dim]
            H: 输入图像 kv 高
            W: 输入图像 kv 宽
            mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(q_n, kv_n, dtype=torch.bool))

        Returns:
        """
        B, N, C = q.shape

        # 获得 Q, K, V 矩阵
        def linear_spatial_reduce(x: torch.Tensor):
            x = x.permute(0, 2, 1).view(B, C, H, W)
            x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)
            return x
        k = linear_spatial_reduce(k)
        v = linear_spatial_reduce(v)

        q, attention = self.multihead_attention(q, k, v, mask=mask)
        return q, attention


if __name__ == '__main__':
    model = LinearSRA(num_heads=8, d_model=24, sr_ratio=2)
    _q = torch.rand(1, 256, 24)     # [B, N, C], q, k, v 的 C 要都一样
    _k = torch.rand(1, 256, 24)
    _v = torch.rand(1, 256, 24)
    _H = 16
    _W = 16
    _out, _attention = model(_q, _k, _v, _H, _W)
    print(_out.shape)
    print(_attention.shape)
