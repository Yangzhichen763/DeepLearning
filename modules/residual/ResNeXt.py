import torch
from torch import nn
from torchvision.models import (
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights
)
from torchvision.models import (
    resnext50_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,
)
from utils.logger import *

from modules.residual.ResNet import BasicBlock as ResNetBasicBlock
from modules.residual.ResNet import num_blocks_dict


class BasicBlock(ResNetBasicBlock):
    """
    ResNeXt 的 BasicBlock 与 ResNet 的 BasicBlock 一样。
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 groups=1, width_per_group=64):
        super(BasicBlock, self).__init__(in_channels, out_channels, stride)


class Bottleneck(nn.Module):
    """
    相比于 ResNet 的 Bottleneck，ResNeXt 的 Bottleneck 增加了分组卷积（也就是 nn.Conv2d 的 groups 参数）。
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channels * (width_per_group / 64)) * groups

        self.feature_layer = nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            # conv3x3
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            # conv1x1
            nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.feature_layer(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNeXtEncoder(nn.Module):
    """
    ResNeXtEncoder is the encoder part of ResNeXt. It consists of several ResNeXt blocks.
    ResNeXtEncoder 是 ResNeXt 的编码器部分，由多个 ResNeXt 块组成。

    输入：
    - shape=[B, N, H, W]

    输出：
    - shape=[B, n, H / l, W / l]，
    其中 l = 2 ** (len(num_blocks) + 1)，n = dim_hidden[-1] * block.expansion，
    如果 len(num_blocks)=4，则 l = 32，shape=[B, n, H / 32, W / 32]。
    """
    def __init__(self, in_channels=3, block=None, num_blocks=None, dim_hidden=None,
                 groups=1, width_per_group=64):
        """
        默认为 ResNet18 的配置，可以根据需要进行修改。
        Args:
            block: 可以选择 BasicBlock、Bottleneck 或者任何自定义的 ResNet 块
            num_blocks: 每个 ResNet 块的数量，默认值为 [2, 2, 2, 2]
            dim_hidden: 隐藏层的维度，默认值为 [64, 128, 256, 512]
        """
        super(ResNeXtEncoder, self).__init__()
        if block is None:
            block = BasicBlock
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if dim_hidden is None:
            dim_hidden = [64, 128, 256, 512]
        strides = [1 if i == 0 else 2 for i in range(len(num_blocks))]
        self.in_channels = dim_hidden[0]
        self.out_channels = dim_hidden[-1] * block.expansion
        self.groups = groups
        self.width_per_group = width_per_group

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, dim_hidden[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dim_hidden[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layers = nn.ModuleList([
            self._make_layer(block, dim_hidden[i], num_blocks[i], strides[i], groups, width_per_group)
            for i in range(len(num_blocks))
        ])
        """ 更直观的写法如下：
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, groups, width_per_group)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, groups, width_per_group)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, groups, width_per_group)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, groups, width_per_group)
        """

        # 初始化权重
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride,
                    groups=1, width_per_group=64):
        strides = [stride] + [1] * (num_blocks - 1)   # 除了第一个 block 其他的都设置 stride=1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, groups, width_per_group))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        for layer in self.layers:
            x = layer(x)
        return x


class ResNeXt(nn.Module):
    """
    Residual Network with Excitation Layers (ResNeXt)
    论文链接 2016：https://arxiv.org/abs/1611.05431
    """
    def __init__(self, in_channels, block, num_blocks, num_classes=10,
                 groups=1, width_per_group=64):
        super(ResNeXt, self).__init__()
        self.encoder = ResNeXtEncoder(in_channels, block, num_blocks, groups=groups, width_per_group=width_per_group)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flattener = nn.Flatten()
        self.linear = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        x = self.flattener(x)
        x = self.linear(x)
        return x


def ResNeXt50_32x4d(in_channels=3, num_classes=10, pretrained=False):
    groups = 32
    width_per_group = 4

    if pretrained:
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        model = resnext50_32x4d(weights=weights)
    else:
        model = ResNeXt(in_channels=in_channels, block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes,
                        groups=groups, width_per_group=width_per_group)
    return model


def ResNeXt101_32x8d(in_channels=3, num_classes=10, pretrained=False):
    groups = 32
    width_per_group = 8

    if pretrained:
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        model = resnext101_32x8d(weights=weights)
    else:
        model = ResNeXt(in_channels=in_channels, block=Bottleneck, num_blocks=num_blocks_dict["ResNet101"], num_classes=num_classes,
                        groups=groups, width_per_group=width_per_group)
    return model


def ResNeXt101_64x4d(in_channels=3, num_classes=10, pretrained=False):
    groups = 64
    width_per_group = 4

    if pretrained:
        weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        model = resnext101_64x4d(weights=weights)
    else:
        model = ResNeXt(in_channels=in_channels, block=Bottleneck, num_blocks=num_blocks_dict["ResNet101"], num_classes=num_classes,
                        groups=groups, width_per_group=width_per_group)
    return model


if __name__ == '__main__':
    # 测试 ResNeXt
    _model = ResNeXt50_32x4d(in_channels=3, num_classes=10, pretrained=False)
    x_input = torch.randn(2, 3, 256, 256)

    log_model_params(_model, x_input.shape)
