import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import enum


"""
参考代码：https://github.com/YuxiangZhang-BIT/IEEE_TIP_SDEnet/blob/main/network/morph_layers2D_torch.py
"""


# Morphological operators
class MorphType(enum.Enum):
    DILATION2D = enum.auto()
    EROSION2D = enum.auto()


class MorphologyConv(nn.Module):
    """
    Base class for morphological operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        """
        in_channels (int): the number of the input channels.
        out_channels (int): the number of the output channels.
        kernel_size (int): the size of morphological kernel.
        soft_max (bool): using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta (int): used by soft_max.
        type (MorphType): dilation-2d or erosion-2d.
        """
        super(MorphologyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.ones(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    # noinspection PyPep8Naming
    def forward(self, x):
        # padding
        # [B, Cin, H, W] -> [B, Cin, H+2p, W+2p], where p = (kernel_size-1)/2
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        # [B, Cin, H+2p, W+2p] -> [B, Cin*k*k, P] -> [B, 1, Cin*k*k, P], where P is the numbers of patches
        x = (self.unfold(x)
             .unsqueeze(1))

        P = x.size(-1)
        P_sqrt = int(math.sqrt(P))
        assert P_sqrt * P_sqrt == P, f'Number of patches {P} is not a square number.'

        # erosion
        # [Cout, Cin, k, k] -> [Cout, Cin*k*k] -> [1, Cout, Cin*k*k, 1]
        weight = (self.weight.view(self.out_channels, -1)
                  .unsqueeze(0).unsqueeze(-1))

        if self.type == MorphType.EROSION2D:
            x = weight - x  # [B, Cout, Cin*k*k, P]
        elif self.type == MorphType.DILATION2D:
            x = weight + x  # [B, Cout, Cin*k*k, P]
        else:
            raise NotImplementedError(f'Morphological type {self.type} is not supported.')

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # [B, Cout, P]
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # [B, Cout, P]

        if self.type == MorphType.EROSION2D:
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, P_sqrt, P_sqrt)  # [B, Cout, P_sqrt, P_sqrt]

        return x


class Dilation2d(MorphologyConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, MorphType.DILATION2D)


class Erosion2d(MorphologyConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, MorphType.EROSION2D)


# Morphological extensions
class MorphExType(enum.Enum):
    OPEN = enum.auto()
    CLOSE = enum.auto()
    GRADIENT = enum.auto()
    TOPHAT = enum.auto()
    BLACKHAT = enum.auto()


class MorphologyExConv(nn.Module):
    def __init__(self):
        super(MorphologyExConv, self).__init__()


class MorphOpen2d(MorphologyExConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(MorphOpen2d, self).__init__()
        self.dilation = Dilation2d(in_channels, out_channels, kernel_size, soft_max, beta)
        self.erosion = Erosion2d(in_channels, out_channels, kernel_size, soft_max, beta)

    def forward(self, x):
        x = self.erosion(x)
        x = self.dilation(x)
        return x


class MorphClose2d(MorphologyExConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(MorphClose2d, self).__init__()
        self.dilation = Dilation2d(in_channels, out_channels, kernel_size, soft_max, beta)
        self.erosion = Erosion2d(in_channels, out_channels, kernel_size, soft_max, beta)

    def forward(self, x):
        x = self.dilation(x)
        x = self.erosion(x)
        return x


class MorphTopHat2d(MorphologyExConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(MorphTopHat2d, self).__init__()
        self.open = MorphOpen2d(in_channels, out_channels, kernel_size, soft_max, beta)

    def forward(self, x):
        x_open = self.open(x)
        x = x - x_open
        return x


class MorphBlackHat2d(MorphologyExConv):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(MorphBlackHat2d, self).__init__()
        self.close = MorphClose2d(in_channels, out_channels, kernel_size, soft_max, beta)

    def forward(self, x):
        x_close = self.close(x)
        x = x_close - x
        return x


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs
