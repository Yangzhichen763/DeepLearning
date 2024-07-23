import torch
import torch.nn as nn
from torchvision.models import (
    densenet121,
    densenet169,
    densenet201,
    densenet161
)
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    DenseNet161_Weights
)
from reproduction.legacy.DenseNet.parts import DenseBlock, TransitionBlock
from utils.log import *


class DenseNet(nn.Module):
    def __init__(self, in_channels, num_channels, growth_rate, num_classes, num_convs_in_dense_blocks=None):
        super(DenseNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        dense_blocks = []
        if num_convs_in_dense_blocks is None:
            num_convs_in_dense_blocks = [4, 4, 4, 4]
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            dense_blocks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_blocks) - 1:
                dense_blocks.append(TransitionBlock(num_channels, num_channels // 2))
                num_channels //= 2
        self.dense_blocks_layer = nn.Sequential(*dense_blocks)

        self.out_conv = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.dense_blocks_layer(x)
        x = self.out_conv(x)
        return x


def DenseNet121(num_classes, in_channels=3, num_channels=64, growth_rate=32, pretrained=False):
    if pretrained:
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights, pretrained=True)
        return model
    else:
        return DenseNet(
            in_channels=in_channels,
            num_channels=num_channels,
            growth_rate=growth_rate,
            num_classes=num_classes,
            num_convs_in_dense_blocks=[6, 12, 24, 16])


def DenseNet169(num_classes, in_channels=3, num_channels=64, growth_rate=32, pretrained=False):
    if pretrained:
        weights = DenseNet169_Weights.IMAGENET1K_V1
        model = densenet169(weights, progress=True)
        return model
    else:
        return DenseNet(
            in_channels=in_channels,
            num_channels=num_channels,
            growth_rate=growth_rate,
            num_classes=num_classes,
            num_convs_in_dense_blocks=[6, 12, 32, 32])


def DenseNet201(num_classes, in_channels=3, num_channels=64, growth_rate=32, pretrained=False):
    if pretrained:
        weights = DenseNet201_Weights.IMAGENET1K_V1
        model = densenet201(weights, progress=True)
        return model
    else:
        return DenseNet(
            in_channels=in_channels,
            num_channels=num_channels,
            growth_rate=growth_rate,
            num_classes=num_classes,
            num_convs_in_dense_blocks=[6, 12, 48, 32])


def DenseNet161(num_classes, in_channels=3, num_channels=96, growth_rate=48, pretrained=False):
    if pretrained:
        weights = DenseNet161_Weights.IMAGENET1K_V1
        model = densenet161(weights, progress=True)
        return model
    else:
        return DenseNet(
            in_channels=in_channels,
            num_channels=num_channels,
            growth_rate=growth_rate,
            num_classes=num_classes,
            num_convs_in_dense_blocks=[6, 12, 36, 24])


if __name__ == '__main__':
    _model = DenseNet(in_channels=3, num_channels=64, growth_rate=32, num_classes=10)
    x_input = torch.randn(1, 3, 32, 32)
    y_pred = _model(x_input)
    print(y_pred.shape)

    log_model_params(_model, input_size=x_input.shape)
