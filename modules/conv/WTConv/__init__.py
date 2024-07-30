import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

import wavelet


"""
WTConv: Wavelet Transform Convolutional Layer
  * 只增加少量可训练参数的情况下，更多地强调输入中的低频部分
  - 使用级联 WT 分解的层，并执行一组小核心卷积，每个卷积在一个越来越大的感受野中专注于输入的不同频带
  - 使用 k × k 的感受野，可训练参数的个数 ∝log(k)
  
小波变换 wavelet transform (WT)
  - 小波分解的意义就在于能够在不同尺度上对信号进行分解，而且对不同尺度的选择可以根据不同的目标来确定
  - 对于许多信号，低频成分相当重要，它常常蕴含着信号的特征，而高频成分则给出信号的细节或差别
  - 在小波分析中经常用到近似与细节。近似表示信号的高尺度，即低频信息；细节表示信号的低尺度，即高频信息。因此，原始信号通过两个相互滤波器产生两个信号
  - 通过不断的分解过程，将近似信号连续分解，就可以将信号分解成许多低分辨率成分
  
论文链接 2024：https://arxiv.org/abs/2407.05848
参考代码：https://github.com/BGU-CS-VIL/WTConv/blob/main/wtconv/wtconv2d.py
"""


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet.wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                       padding='same', stride=1, dilation=1, groups=in_channels * 4, bias=False)
             for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
             for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter,
                                                   bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class ScaleModule(nn.Module):
    """
    逐点权重缩放模块
    """
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


class LinearModule(nn.Module):
    """
    逐点权重线性模块
    """
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super().__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = nn.Parameter(torch.ones(*dims) * init_bias)

    def forward(self, x):
        return torch.mul(self.weight, x) + self.bias


if __name__ == '__main__':
    from utils.log.model import log_model_params

    x_input = torch.randn(2, 3, 16, 16)
    wt_conv = WTConv2d(3, 3, 3, 1, True, 1, 'db1')

    log_model_params(wt_conv, input_data=x_input)
