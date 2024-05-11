import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights
)
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)
from utils.logger import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.feature_layer(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),    # 后面接 BN，所以 bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.feature_layer(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flattener = nn.Flatten()
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)   # 除了第一个 block 其他的都设置 stride=1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.flattener(x)
        x = self.linear(x)
        return x


def ResNet18(num_classes=10, pretrained=False):
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        return model
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10, pretrained=False):
    if pretrained:
        weights = ResNet34_Weights.IMAGENET1K_V1
        model = resnet34(weights=weights)
        return model
    else:
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10, pretrained=False):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        return model
    else:
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=10, pretrained=False):
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V1
        model = resnet101(weights=weights)
        return model
    else:
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=10, pretrained=False):
    if pretrained:
        return models.resnet152(pretrained=True)
    else:
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == '__main__':
    _model = ResNet34()
    x_input = torch.randn(1, 3, 32, 32)
    y_pred = _model(x_input)
    print(y_pred.shape)

    log_model_params(_model, x_input.shape)
