import torch
import torch.nn as nn
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


"""
Residual Network (ResNet)
论文链接 2015：https://arxiv.org/abs/1512.03385
"""


num_blocks_dict = {
    "ResNet18": [2, 2, 2, 2],
    "ResNet34": [3, 4, 6, 3],
    "ResNet50": [3, 4, 6, 3],
    "ResNet101": [3, 4, 23, 3],
    "ResNet152": [3, 8, 36, 3]
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        activation = kwargs["activation"] if kwargs.get("activation") is not None else nn.ReLU(inplace=True)

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.activation = activation

    def forward(self, x):
        out = self.feature_layer(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(Bottleneck, self).__init__()
        activation = kwargs["activation"] if kwargs.get("activation") is not None else nn.ReLU(inplace=True)

        self.feature_layer = nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),    # 后面接 BN，所以 bias=False
            nn.BatchNorm2d(out_channels),
            activation,
            # conv3x3
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            # conv1x1
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.activation = activation

    def forward(self, x):
        out = self.feature_layer(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNetEncoder(nn.Module):
    """
    ResNetEncoder is the encoder part of ResNet. It consists of several ResNet blocks.
    ResNetEncoder 是 ResNet 的编码器部分，由多个 ResNet 块组成。

    输入：
    - shape=[B, N, H, W]

    输出：
    - shape=[B, n, H / l, W / l]，
    其中 l = 2 ** (len(num_blocks) + 1)，n = dim_hidden[-1] * block.expansion，
    如果 len(num_blocks)=4，则 l = 32，shape=[B, n, H / 32, W / 32]。
    """
    def __init__(self, in_channels=3, block=None, num_blocks=None, dim_hidden=None, **kwargs):
        """
        默认为 ResNet18 的配置，可以根据需要进行修改。
        Args:
            block: 可以选择 BasicBlock、Bottleneck 或者任何自定义的 ResNet 块
            num_blocks: 每个 ResNet 块的数量，默认值为 [2, 2, 2, 2]
            dim_hidden: 隐藏层的维度，默认值为 [64, 128, 256, 512]
        """
        super(ResNetEncoder, self).__init__()
        activation = kwargs["activation"] if kwargs.get("activation") is not None else nn.ReLU(inplace=True)

        if block is None:
            block = BasicBlock
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if dim_hidden is None:
            dim_hidden = [64, 128, 256, 512]
        strides = [1 if i == 0 else 2 for i in range(len(num_blocks))]
        self.in_channels = dim_hidden[0]
        self.out_channels = dim_hidden[-1] * block.expansion

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, dim_hidden[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dim_hidden[0]),
            activation,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layers = nn.ModuleList([
            self._make_layer(block, dim_hidden[i], num_blocks[i], strides[i], **kwargs)
            for i in range(len(num_blocks))
        ])
        """ 更直观的写法如下：
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        """

        # 初始化权重
        self.init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)   # 除了第一个 block 其他的都设置 stride=1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, **kwargs))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained))
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.layer0(x)
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(nn.Module):
    """
    Residual Network (ResNet)
    论文链接 2015：https://arxiv.org/abs/1512.03385
    """
    def __init__(self, in_channels=3, block=None, num_blocks=None, num_classes=10):
        """

        Args:
            in_channels:
            block: BasicBlock、Bottleneck 或者自定义的 ResNet 块
            num_blocks: 每一层的 block 数量
            num_classes:
        """
        super(ResNet, self).__init__()
        self.encoder = ResNetEncoder(in_channels, block, num_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flattener = nn.Flatten()
        self.linear = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        x = self.flattener(x)
        x = self.linear(x)
        return x


def ResNet18(pretrained=False, **kwargs):
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        return model
    else:
        return ResNet(block=BasicBlock, num_blocks=num_blocks_dict["ResNet18"], **kwargs)


def ResNet34(pretrained=False, **kwargs):
    if pretrained:
        weights = ResNet34_Weights.IMAGENET1K_V1
        model = resnet34(weights=weights)
        return model
    else:
        return ResNet(block=BasicBlock, num_blocks=num_blocks_dict["ResNet34"], **kwargs)


def ResNet50(pretrained=False, **kwargs):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        return model
    else:
        return ResNet(block=Bottleneck, num_blocks=num_blocks_dict["ResNet50"], **kwargs)


def ResNet101(pretrained=False, **kwargs):
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V1
        model = resnet101(weights=weights)
        return model
    else:
        return ResNet(block=Bottleneck, num_blocks=num_blocks_dict["ResNet101"], **kwargs)


def ResNet152(pretrained=False, **kwargs):
    if pretrained:
        weights = ResNet152_Weights.IMAGENET1K_V1
        model = resnet152(weights=weights)
        return model
    else:
        return ResNet(block=Bottleneck, num_blocks=num_blocks_dict["ResNet152"], **kwargs)


if __name__ == '__main__':
    from utils.log import log_model_params

    x_input = torch.randn(2, 3, 32 * 7, 32 * 7)
    _model = ResNet34()

    log_model_params(_model, input_data=x_input)
