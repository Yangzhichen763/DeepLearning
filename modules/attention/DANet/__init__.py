import torch
import torch.nn as nn
from utils.torch.nn import flatten_     # flatten_ 相当于 view，但是更加灵活


"""
DANet: Dual Attention Network for Scene Segmentation
DANet 结合了多种注意力机制，可以同时学习全局和局部的上下文信息。
论文链接 2018-2019：https://arxiv.org/abs/1809.02983
"""


class ChannelAttention(nn.Module):
    """
    DANet 中的通道注意力模块
    """
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    # noinspection PyTestUnpassedFixture
    def forward(self, x: torch.Tensor):
        q = x.flatten_(start_dim=-2)
        k = x.flatten_(start_dim=-2).transpose(-1, -2)
        attn = torch.bmm(q, k)     # torch.bmm 在处理批量矩阵乘法的性能比 torch.matmul（或者 @ 运算符）要好
        attn = torch.max(attn, dim=-1, keepdim=True)[0].expand_as(attn) - attn
        attn = self.softmax(attn)
        v = x.flatten_(start_dim=-2)

        y = torch.bmm(attn, v)
        y = y.view_as(x)

        y = self.gamma * y + x
        return y


class PositionAttention(nn.Module):
    """
    DANet 中的位置注意力模块
    """
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.in_channels = in_channels

        self.q_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    # noinspection PyTestUnpassedFixture
    def forward(self, x: torch.Tensor):
        q = self.q_conv(x).flatten_(start_dim=-2).transpose(-1, -2)
        k = self.k_conv(x).flatten_(start_dim=-2)
        attn = torch.bmm(q, k)
        attn = self.softmax(attn)
        v = self.v_conv(x).flatten_(start_dim=-2)

        y = torch.bmm(v, attn.transpose(-1, -2))
        y = y.view_as(x)

        y = self.gamma * y + x
        return y


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(DANetHead, self).__init__()
        self.in_channels = in_channels // 4
        self.out_channels = out_channels
        self.dropout = dropout

        self.conv_layer_c = nn.Sequential(
            self.make_conv_layer(in_channels, in_channels),
            ChannelAttention(in_channels),
            self.make_conv_layer(in_channels, in_channels),
        )
        self.dropout_conv_c = self.make_dropout_conv_layer(in_channels, out_channels)

        self.conv_layer_p = nn.Sequential(
            self.make_conv_layer(in_channels, in_channels),
            PositionAttention(in_channels),
            self.make_conv_layer(in_channels, in_channels),
        )
        self.dropout_conv_p = self.make_dropout_conv_layer(in_channels, out_channels)

        self.sum_conv = self.make_dropout_conv_layer(in_channels, out_channels)

    def make_conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

#
    def make_dropout_conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Dropout2d(self.dropout, inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        c = self.conv_layer_c(x)        # [B, C, H, W] -> [B, C, H, W]
        c_out = self.dropout_conv_c(c)  # [B, C, H, W] -> [B, 1, H, W]

        p = self.conv_layer_p(x)        # [B, C, H, W] -> [B, C, H, W]
        p_out = self.dropout_conv_p(p)  # [B, C, H, W] -> [B, 1, H, W]

        y = c + p
        y = self.sum_conv(y)            # [B, C, H, W] -> [B, 1, H, W]
        return y, c_out, p_out


if __name__ == '__main__':
    from utils.log import log_model_params

    # 测试 ChannelAttention
    print("\nChannelAttention:")
    x_input = torch.randn(2, 64, 16, 16)
    model = ChannelAttention(x_input.shape[1])

    log_model_params(model, input_data=x_input)

    # 测试 PositionAttention
    print("\nPositionAttention:")
    model = PositionAttention(x_input.shape[1])

    log_model_params(model, input_data=x_input)

    # 测试 DANetHead
    print("\nDANetHead:")
    model = DANetHead(x_input.shape[1], 1)

    log_model_params(model, input_data=x_input)
