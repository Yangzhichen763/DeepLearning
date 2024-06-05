import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

from utils.logger import log_model_params


"""
Res2Net: A New Multi-scale Backbone Architecture
论文连接 2019-2021：https://arxiv.org/abs/1904.01169
"""


__all__ = ['Res2Net', 'res2net50']


model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}


class StageConv(nn.Module):
    """
    Res2Net 位于千 1x1 卷积和后 1x1 卷积之间的中间模块
    """
    def __init__(self, in_channels, out_channels, stride=1, scale=4, stype='normal'):
        """

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
            scale (int):
            stype: 如果是
        """
        super(StageConv, self).__init__()
        self.width = in_channels
        self.scale = scale
        self.stype = stype

        # 分块剩余部分的中间模块，为直达块或特殊块
        if stype == 'stage':
            special_block = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            special_block = nn.Sequential()

        # 分块卷积（其中前 scale-1 个为普通卷积，最后一个为直达块或特殊块）
        self.convs = nn.ModuleList([
            special_block
            if i == self.scale - 1 else
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for i in range(self.scale)
        ])

    def forward(self, x):
        x_splits = torch.split(x, self.width, dim=1)

        x_out = []
        if self.stype == 'stage':      # Inception-like stage
            for i in range(0, self.scale):
                y = x_splits[i]
                y = self.convs[i](y)
                x_out.append(y)
            x = torch.cat(x_out, 1)
        else:                                   # Res2Net block
            x_out.append(self.convs[0](x_splits[0]))
            for i in range(1, self.scale):
                y = x_out[i - 1] + x_splits[i]
                y = self.convs[i](y)
                x_out.append(y)
            x = torch.cat(x_out, 1)
        return x


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, base_width=26, scale=4, stype='normal'):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长，使用步长代替池化操作
            base_width: conv3x3 块的通道数
            scale: 分块数
            stype: 'normal': 普通设置. 'stage': 下采样时使用 avg pool 代替 3x3 卷积.
        """
        super(Bottle2neck, self).__init__()

        self.stype = stype
        self.scale = scale
        self.width = math.floor(out_channels * (base_width / 64.0))

        self.feature_layer = nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels, self.width * scale, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.width * scale),
            nn.ReLU(inplace=True),
            # conv3x3
            StageConv(self.width, self.width, stride, scale, stype=stype),
            # nn.ReLU(inplace=True),  # 原论文没有这个激活函数
            # conv1x1
            nn.Conv2d(self.width * scale, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
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


class Res2NetEncoder(nn.Module):
    """
    Res2NetEncoder is the encoder part of Res2Net. It consists of several Res2Net blocks.
    Res2NetEncoder 是 Res2Net 的编码器部分，由多个 Res2Net 块组成。

    输入：
    - shape=[B, N, H, W]

    输出：
    - shape=[B, n, H / l, W / l]，
    其中 l = 2 ** (len(num_blocks) + 1)，n = dim_hidden[-1] * block.expansion，
    如果 len(num_blocks)=4，则 l = 32，shape=[B, n, H / 32, W / 32]。
    """
    def __init__(self,
                 in_channels=3, block=None, num_blocks=None, dim_hidden=None,
                 base_width=26, scale=4):
        """
        默认为 Res2Net18 的配置，可以根据需要进行修改。
        Args:
            block: 可以选择 BasicBlock、Bottleneck 或者任何自定义的 ResNet 块
            num_blocks: 每个 ResNet 块的数量，默认值为 [2, 2, 2, 2]
            dim_hidden: 隐藏层的维度，默认值为 [64, 128, 256, 512]
        """
        super(Res2NetEncoder, self).__init__()
        if block is None:
            block = Bottle2neck
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if dim_hidden is None:
            dim_hidden = [64, 128, 256, 512]
        strides = [1 if i == 0 else 2 for i in range(len(num_blocks))]
        self.in_channels = dim_hidden[0]
        self.out_channels = dim_hidden[-1] * block.expansion
        self.base_width = base_width
        self.scale = scale

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, dim_hidden[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dim_hidden[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layers = nn.ModuleList([
            self._make_layer(block, dim_hidden[i], num_blocks[i], strides[i])
            for i in range(len(num_blocks))
        ])
        """ 更直观的写法如下：
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        """

        # 初始化权重
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)         # 除了第一个 block 其他的都设置 stride=1
        stypes = ['stage'] + ['normal'] * (num_blocks - 1)  # 除了第一个 block 其他的都设置 stype='normal'
        layers = []
        for (stype, stride) in zip(stypes, strides):
            layers.append(block(self.in_channels, out_channels, stride=stride,
                                base_width=self.base_width, scale=self.scale, stype=stype))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Res2Net(nn.Module):
    """
    Res2Net
    论文链接：https://arxiv.org/abs/1904.01169
    """
    def __init__(self, in_channels=3, block=None, num_blocks=None, base_width=26, scale=4, num_classes=10):
        """

        Args:
            in_channels:
            block: BasicBlock、Bottleneck 或者自定义的 ResNet 块
            num_blocks: 每一层的 block 数量
            num_classes:
        """
        super(Res2Net, self).__init__()
        self.encoder = Res2NetEncoder(in_channels, block, num_blocks,
                                      base_width=base_width, scale=scale)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flattener = nn.Flatten()
        self.linear = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        x = self.flattener(x)
        x = self.linear(x)
        return x


def res2net50(pretrained=False, **kwargs):
    """
    Res2Net-50 也就是 Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = Res2Net(block=Bottle2neck, num_blocks=[3, 4, 6, 3], base_width=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net50_26w_4s(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = Res2Net(block=Bottle2neck, num_blocks=[3, 4, 6, 3], base_width=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net101_26w_4s(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = Res2Net(block=Bottle2neck, num_blocks=[3, 4, 23, 3], base_width=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model


def res2net50_26w_6s(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = Res2Net(block=Bottle2neck, num_blocks=[3, 4, 6, 3], base_width=26, scale=6, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model


def res2net50_26w_8s(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = Res2Net(block=Bottle2neck, num_blocks=[3, 4, 6, 3], base_width=26, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    return model


def res2net50_48w_2s(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = Res2Net(block=Bottle2neck, num_blocks=[3, 4, 6, 3], base_width=48, scale=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model


def res2net50_14w_8s(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = Res2Net(block=Bottle2neck, num_blocks=[3, 4, 6, 3], base_width=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_14w_8s']))
    return model


if __name__ == '__main__':
    x_input = torch.rand(1, 3, 224, 224)
    _model = res2net101_26w_4s()

    log_model_params(_model, x_input.shape)
