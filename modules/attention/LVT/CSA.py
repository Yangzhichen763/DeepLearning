import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import sys
sys.path.append('..')
# noinspection PyUnresolvedReferences
from Attention import MultiheadAttention, SelfAttention
sys.path.pop()


"""
LVT: Lite Vision Transformer
  - 对于 Low-level 特征，引入了卷积子注意力（CSA），将局部注意力引入到大小为 3x3 卷积中，以丰富 Low-level 特征。
论文链接：https://arxiv.org/abs/2112.10809
参考代码：https://github.com/Chenglin-Yang/LVT/blob/main/classification/models/lvt.py
"""


class CSA(nn.Module):
    """
    Convolutional Self-Attention (CSA) module.
    """
    def __init__(self, *, in_channels, out_channels,
                 num_heads, qk_scale=None,
                 kernel_size=3, padding=1, stride=1,    # 卷积参数
                 csa_groups=1,                          # 卷积注意力组数
                 dropout=0.1):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_heads (int): 多头注意力头数
            qk_scale (float): 注意力机制的缩放因子
            kernel_size (int): 卷积核大小
            padding (int): 卷积填充
            stride (int): 卷积步长
            csa_groups (int): 卷积注意力组数，默认且最好为 1
            dropout (float): dropout概率
        """
        super(CSA, self).__init__()
        assert in_channels == out_channels, "in_channels must equal to out_channels"
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.in_channels = in_channels
        self.out_channels = out_channels
        head_dim = out_channels // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim ** -0.5

        # 对输入 q, k, v 进行全连接
        self.weight_qkv = nn.Linear(in_channels, kernel_size ** 4 * num_heads, bias=False)
        self.dropout_qkv = nn.Dropout(dropout)

        # 卷积注意力
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        assert out_channels % csa_groups == 0, "out_channels must be divisible by csa_groups"
        self.weight_v = nn.Conv2d(
            self.kernel_size * self.kernel_size * out_channels,
            self.kernel_size * self.kernel_size * out_channels,
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=self.kernel_size * self.kernel_size * csa_groups, bias=False
        )

        # 权重初始化 init_weights
        fan_out = self.kernel_size * self.kernel_size * out_channels // csa_groups
        self.weight_v.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

        # 对输出进行全连接
        self.projection = nn.Linear(out_channels, out_channels, bias=False)
        self.dropout_out = nn.Dropout(dropout)

    # noinspection PyPep8Naming
    def forward(self, x):
        """
        Args:
            x (Tensor): [B, H, W, C]
        """
        B, H, W, C = x.shape
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        kernel_size_sqr = self.kernel_size ** 2

        # 对输入 q, k 进行全连接
        attention = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # -> [B, Cin, h, w] -> [B, h, w, Cin]
        attention = (
            self.weight_qkv(attention)                                             # -> [B, h, w, N*k**4], N=num_heads
            .reshape(B, h * w, self.num_heads, kernel_size_sqr, kernel_size_sqr)   # -> [B, h*w, N, k*k, k*k]
            .permute(0, 2, 1, 3, 4))                                               # -> [B, N, h*w, k*k, k*k]
        attention = attention * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout_qkv(attention)

        # 对输入 v 进行卷积注意力
        v = x.permute(0, 3, 1, 2)   # [B, Cin, H, W] -> [B, H, W, Cin]
        v = (
            self.unfold(v)                                                      # -> [B, Cin*k*k, H*W]
            .reshape(B, self.out_channels, kernel_size_sqr, h * w)              # -> [B, Cout, k*k, h*w]
            .permute(0, 3, 2, 1)                                                # -> [B, h*w, k*k, Cout]
            .reshape(B * h * w, kernel_size_sqr * self.out_channels, 1, 1))     # -> [B*h*w, k*k*Cout, 1, 1]
        v = self.weight_v(v)                                                    # --
        # -> [B, h*w, k*k, N, C/N] -> [B, N, h*w, k*k, C/N]
        v = (v
             .reshape(B, h * w, kernel_size_sqr, self.num_heads, self.out_channels // self.num_heads)
             .permute(0, 3, 1, 2, 4)
             .contiguous())

        # 将 v 与 attention 相乘
        # -> [B, N, h*w, k*k, k*k] x [B, N, h*w, k*k, C/N] -> [B, N, h*w, k*k, C/N]
        x = (attention @ v).permute(0, 1, 4, 3, 2)
        x = x.reshape(B, self.out_channels * kernel_size_sqr, h * w)    # -> [B, Cout*k*k, h*w]
        x = F.fold(x,                                                   # -> [B, Cout, H, W]
                   output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        # 对输出进行全连接
        x = self.projection(x.permute(0, 2, 3, 1))                # -> [B, H, W, Cout]
        x = self.dropout_out(x)
        return x


if __name__ == '__main__':
    from utils.log import log_model_params

    x_input = torch.randn(2, 32, 32, 128)
    model = CSA(in_channels=128, out_channels=128, num_heads=4, kernel_size=3, padding=1, stride=1, csa_groups=1)
    log_model_params(model, input_data=x_input)
