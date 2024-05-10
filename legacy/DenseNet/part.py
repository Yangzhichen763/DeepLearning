import torch
from torch import nn


def Conv2dBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )


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
