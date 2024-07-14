import torch
from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2dBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(Conv2dBlock(in_channels + i * out_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for block in self.net:
            y = block(x)
            x = torch.cat((x, y), dim=1)    # 连接通道维度上每个块的输入和输出
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.net(x)
