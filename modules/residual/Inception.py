import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


__all__ = ['Inception_v1',
           'Inception_v2_A', 'Inception_v2_B', 'Inception_v2_C',
           'Inception_v4']


class BasicConv2d(nn.Module):
    """
    Basic convolution block with batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Args:
            in_channels:
            out_channels:
            **kwargs: nn.Conv2d 的其他参数
        """
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Inception_v1(nn.Module):
    """
    Inception module with dimension reduction
    论文链接 2014：https://arxiv.org/abs/1409.4842
    """
    def __init__(self, in_channels, dim_1x1=None, dim_3x3=None, dim_5x5=None, dim_pool=None):
        """
        Args:
            in_channels:
            dim_1x1:        默认值为 [64]               -> 1x1
            dim_3x3:        默认值为 [96, 128]          -> 1x1, 3x3
            dim_5x5:        默认值为 [16, 32]           -> 1x1, 5x5
            dim_pool:       默认值为 [in_channels, 32]  -> pool, 1x1
        """
        super(Inception_v1, self).__init__()
        if dim_1x1 is None:
            dim_1x1 = [64]
        if dim_3x3 is None:
            dim_3x3 = [96, 128]
        if dim_5x5 is None:
            dim_5x5 = [16, 32]
        if dim_pool is None:
            dim_pool = [in_channels, 32]

        self.branch_1x1 = nn.Sequential(
            BasicConv2d(in_channels, dim_1x1[0], kernel_size=1, stride=1, padding=0)
        )

        self.branch_3x3 = nn.Sequential(
            BasicConv2d(in_channels, dim_3x3[0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(dim_3x3[0], dim_3x3[1], kernel_size=3, stride=1, padding=1)
        )

        self.branch_5x5 = nn.Sequential(
            BasicConv2d(in_channels, dim_5x5[0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(dim_5x5[0], dim_5x5[1], kernel_size=5, stride=1, padding=2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(dim_pool[0], dim_pool[1], kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_5x5 = self.branch_5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch_1x1, branch_3x3, branch_5x5, branch_pool]
        return torch.cat(outputs, 1)


class ConvBlock_1xn_nx1(nn.Module):
    """
    Convolution block with 1x3 and 3x1 filters
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=-1):
        super(ConvBlock_1xn_nx1, self).__init__()
        half_out_channels = out_channels // 2
        if padding == -1:
            padding = kernel_size // 2

        self.conv_1xn = BasicConv2d(in_channels, half_out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding))
        self.conv_nx1 = BasicConv2d(in_channels, half_out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0))

    def forward(self, x):
        conv_1xn = self.conv_1xn(x)
        conv_nx1 = self.conv_nx1(x)
        return torch.cat([conv_1xn, conv_nx1], 1)


class ConvBlock_1x3_3x1(ConvBlock_1xn_nx1):
    """
    Convolution block with 1x3 and 3x1 filters
    """
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ConvBlock_1x3_3x1, self).__init__(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)


class Inception_v2_A(nn.Module):
    """
    Inception modules where each 5x5 convolution is replaces by two 3x3 convolution
    论文链接 2015：https://arxiv.org/abs/1512.00567
    """
    def __init__(self,
                 in_channels,
                 dim_1x1=None,
                 dim_pool=None,
                 dim_3x3=None,
                 dim_double3x3=None):
        """
        Args:
            in_channels:
            dim_1x1:        默认值为 [32]               -> 1x1
            dim_pool:       默认值为 [in_channels, 32]  -> pool, 1x1
            dim_3x3:        默认值为 [96, 128]          -> 1x1, 1x3&3x1
            dim_double3x3:        默认值为 [32, 64, 64]       -> 1x1, 3x3, 1x3&3x1
        """
        super(Inception_v2_A, self).__init__()
        if dim_1x1 is None:
            dim_1x1 = [96]
        if dim_pool is None:
            dim_pool = [in_channels, 96]
        if dim_3x3 is None:
            dim_3x3 = [64, 96]
        if dim_double3x3 is None:
            dim_double3x3 = [64, 96, 96]

        self.branch_1x1 = nn.Sequential(
            BasicConv2d(in_channels, dim_1x1[0], kernel_size=1, stride=1, padding=0)
        )

        self.branch_pool_1x1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(dim_pool[0], dim_pool[1], kernel_size=1, stride=1, padding=0)
        )

        self.branch_3x3 = nn.Sequential(
            BasicConv2d(in_channels, dim_3x3[0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(dim_3x3[0], dim_3x3[1], kernel_size=3, stride=1, padding=1),
        )

        self.branch_double_3x3 = nn.Sequential(
            BasicConv2d(in_channels, dim_double3x3[0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(dim_double3x3[0], dim_double3x3[1], kernel_size=3, stride=1, padding=1),
            BasicConv2d(dim_double3x3[1], dim_double3x3[2], kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_double_3x3 = self.branch_double_3x3(x)
        branch_pool = self.branch_pool_1x1(x)

        outputs = [branch_1x1, branch_3x3, branch_double_3x3, branch_pool]
        return torch.cat(outputs, 1)


class Inception_v2_B(nn.Module):
    """
    Inception modules where each 5x5 convolution is replaces by two 3x3 convolution
    论文链接 2015：https://arxiv.org/abs/1512.00567
    """
    def __init__(self,
                 in_channels,
                 n=7,
                 dim_1x1=None,
                 dim_pool=None,
                 dim_1xn_nx1=None,
                 dim_double_1xn_nx1=None):
        """
        Args:
            in_channels:
            dim_1x1:            默认值为 [32]               -> 1x1
            dim_pool:           默认值为 [in_channels, 32]  -> pool, 1x1
            dim_1xn_nx1:        默认值为 [96, 128]          -> 1x1, 1xn&nx1
            dim_double_1xn_nx1: 默认值为 [32, 64, 64]       -> 1x1, 1xn&nx1, 1xn&nx1
        """
        super(Inception_v2_B, self).__init__()
        if dim_1x1 is None:
            dim_1x1 = [384]
        if dim_pool is None:
            dim_pool = [in_channels, 128]
        if dim_1xn_nx1 is None:
            dim_1xn_nx1 = [192, 224, 256]
        if dim_double_1xn_nx1 is None:
            dim_double_1xn_nx1 = [192, 192, 224, 224, 256]

        self.branch1x1 = nn.Sequential(
            BasicConv2d(in_channels, dim_1x1[0], kernel_size=1, stride=1, padding=0)
        )

        self.branch_pool_1x1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(dim_pool[0], dim_pool[1], kernel_size=1, stride=1, padding=0)
        )

        self.branch_1xn_nx1 = nn.Sequential(
            BasicConv2d(in_channels, dim_1xn_nx1[0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(dim_1xn_nx1[0], dim_1xn_nx1[1], kernel_size=(1, n), stride=1, padding=(0, n // 2)),
            BasicConv2d(dim_1xn_nx1[1], dim_1xn_nx1[2], kernel_size=(n, 1), stride=1, padding=(n // 2, 0))
        )

        self.branch_double_1xn_nx1 = nn.Sequential(
            BasicConv2d(in_channels, dim_double_1xn_nx1[0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(dim_1xn_nx1[0], dim_1xn_nx1[1], kernel_size=(1, n), stride=1, padding=(0, n // 2)),
            BasicConv2d(dim_1xn_nx1[1], dim_1xn_nx1[2], kernel_size=(n, 1), stride=1, padding=(n // 2, 0)),
            BasicConv2d(dim_1xn_nx1[2], dim_1xn_nx1[3], kernel_size=(1, n), stride=1, padding=(0, n // 2)),
            BasicConv2d(dim_1xn_nx1[3], dim_1xn_nx1[4], kernel_size=(n, 1), stride=1, padding=(n // 2, 0))
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch_1xn_nx1 = self.branch_1xn_nx1(x)
        branch_double_1xn_nx1 = self.branch_double_1xn_nx1(x)
        branch_pool = self.branch_pool_1x1(x)

        outputs = [branch1x1, branch_1xn_nx1, branch_double_1xn_nx1, branch_pool]
        return torch.cat(outputs, 1)


class Inception_v2_C(nn.Module):
    """
    Inception modules with expanded the filter bank outputs
    论文链接 2015：https://arxiv.org/abs/1512.00567
    """
    def __init__(self, in_channels, dim_1x1=None, dim_pool=None, dim_3x3=None, dim_5x5=None):
        """
        Args:
            in_channels:
            dim_1x1:        默认值为 [32]               -> 1x1
            dim_pool:       默认值为 [in_channels, 32]  -> pool, 1x1
            dim_3x3:        默认值为 [96, 128]          -> 1x1, 1x3&3x1
            dim_5x5:        默认值为 [32, 64, 64]       -> 1x1, 3x3, 1x3&3x1
        """
        super(Inception_v2_C, self).__init__()
        if dim_1x1 is None:
            dim_1x1 = [256]
        if dim_pool is None:
            dim_pool = [in_channels, 256]
        if dim_3x3 is None:
            dim_3x3 = [384, 256]
        if dim_5x5 is None:
            dim_5x5 = [384, 512, 256]

        self.branch_1x1 = nn.Sequential(
            BasicConv2d(in_channels, dim_1x1[0], kernel_size=1, stride=1, padding=0)
        )

        self.branch_pool_1x1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(dim_pool[0], dim_pool[1], kernel_size=1, stride=1, padding=0)
        )

        self.branch_3x3 = nn.Sequential(
            BasicConv2d(in_channels, dim_3x3[0], kernel_size=1, stride=1, padding=0),
            ConvBlock_1x3_3x1(dim_3x3[0], dim_3x3[1]),
        )

        self.branch_5x5 = nn.Sequential(
            BasicConv2d(in_channels, dim_5x5[0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(dim_5x5[0], dim_5x5[1], kernel_size=3, stride=1, padding=1),
            ConvBlock_1x3_3x1(dim_5x5[1], dim_5x5[2])
        )

    def forward(self, x):
        branch1x1 = self.branch_1x1(x)
        branch3x3 = self.branch_3x3(x)
        branch5x5 = self.branch_5x5(x)
        branch_pool = self.branch_pool_1x1(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class Inception_v4(nn.Module):
    """
    Inception-v4 module
    论文链接 2016：https://arxiv.org/abs/1602.07261
    论文中有多种 Inception 模块以及 Inception-ResNet 模块，就不一一实现了
    """
    pass


if __name__ == '__main__':
    x_input = torch.randn(1, 3, 224, 224)
    _in_channels = x_input.shape[1]

    models = [Inception_v1, Inception_v2_A, Inception_v2_B, Inception_v2_C]
    for model in models:
        inception = model(_in_channels)
        y_output = inception(x_input)
        print(f"{model.__name__}: {y_output.shape}")

