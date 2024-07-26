
import enum
import torch.nn as nn

"""
StarNet 能够在低维空间中计算，同时产生高维特征
  - 网络结构极简，显著减少人工干预，性能卓越
论文链接 2024：https://arxiv.org/abs/2403.19967
代码参考：https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py
"""


class OperatorMode(enum.Enum):
    SUM = enum.auto()
    STAR = enum.auto()


class ConvBlock(nn.Module):
    """
    卷积块，包含卷积层和 BN 层
    """
    def __init__(self, in_channels, out_channels, with_bn=True, **conv_kwargs):
        super().__init__()

        self.dw_conv = nn.Conv2d(in_channels, out_channels, **conv_kwargs)

        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

            self.net = nn.Sequential(
                self.dw_conv,
                self.norm
            )
        else:
            self.net = nn.Sequential(
                self.dw_conv,
            )

    def forward(self, x):
        x = self.net(x)
        return x


class StarBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 hidden_dim: int,
                 mode: OperatorMode = OperatorMode.STAR,
                 activation: nn.Module = nn.ReLU6()):
        super().__init__()

        self.mode = mode

        self.in_dw_conv = ConvBlock(channels, channels,
                                    kernel_size=7, stride=1, padding=3, groups=channels, with_bn=True)
        self.in_f1 = ConvBlock(channels, hidden_dim, kernel_size=1, with_bn=False)
        self.in_f2 = ConvBlock(channels, hidden_dim, kernel_size=1, with_bn=False)
        self.in_activation = activation

        self.out_g = ConvBlock(hidden_dim, channels, kernel_size=1, with_bn=True)
        self.out_dw_conv = ConvBlock(channels, channels,
                                     kernel_size=7, stride=1, padding=3, groups=channels, with_bn=False)

    def forward(self, x):
        x_residual = x

        x = self.in_dw_conv(x)
        x1, x2 = self.in_f1(x), self.in_f2(x)
        if self.mode == OperatorMode.SUM:
            x = self.in_activation(x1) + x2
        elif self.mode == OperatorMode.STAR:
            x = self.in_activation(x1) * x2
        else:
            raise NotImplementedError(f"Unsupported operator mode: {self.mode}")

        x = self.out_g(x)
        x = self.out_dw_conv(x)

        x = x + x_residual
        return x


if __name__ == '__main__':
    import torch
    from utils.log.model import log_model_params

    x_channels = 3
    x_input = torch.randn(2, x_channels, 224, 224)
    model = StarBlock(channels=x_channels, hidden_dim=32, mode=OperatorMode.STAR)
    log_model_params(model, input_data=x_input)
