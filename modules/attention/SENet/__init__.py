import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.residual.ResNet import ResNet
from utils.logger import log_model_params


"""
SENet: Squeeze-and-Excitation Networks
SENet 是一种通道注意力机制 CAM(Channel Attention Module)
论文链接 2017-2019：https://arxiv.org/abs/1709.01507
"""


class SEBlock(nn.Module):
    """
    更泛的通道注意力机制见 CBAM(Convolutional Block Attention Module)
    """
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.se_layer = SEBlock(out_channels)   # 相对于普通的 ResNet，添加这行代码(SE 模块)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.feature_layer(x)
        out = self.se_layer(out)    # 相对于普通的 ResNet，添加这行代码(SE 模块)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.feature_layer = nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),    # 后面接 BN，所以 bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # conv3x3
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # conv1x1
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.se_layer = SEBlock(out_channels)   # 相对于普通的 ResNet，添加这行代码(SE 模块)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.feature_layer(x)
        out = self.se_layer(out)    # 相对于普通的 ResNet，添加这行代码(SE 模块)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SEResNet(ResNet):
    def __init__(self, in_channels=3, block=None, num_blocks=None, num_classes=10):
        super(SEResNet, self).__init__(
            in_channels=in_channels,
            block=block,
            num_blocks=num_blocks,
            num_classes=num_classes)


def SEResNet18(**kwargs):
    return SEResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], **kwargs)


def SEResNet34(**kwargs):
    return SEResNet(block=BasicBlock, num_blocks=[3, 4, 6, 3], **kwargs)


def SEResNet50(**kwargs):
    return SEResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], **kwargs)


def SEResNet101(**kwargs):
    return SEResNet(block=Bottleneck, num_blocks=[3, 4, 23, 3], **kwargs)


def SEResNet152(**kwargs):
    return SEResNet(block=Bottleneck, num_blocks=[3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    x_input = torch.randn(1, 3, 224, 224)
    model = SEResNet18()

    log_model_params(model, x_input.shape)
