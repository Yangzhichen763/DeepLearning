import torch
from torch import nn
import numpy as np


class NiN(nn.Module):
    def __init__(self, num_classes=1000):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            make_nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            make_nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            make_nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            make_nin_block(384, num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()    # 将四维的输出转成二维的输出，其形状为 [batch_size, num_classes]
        )

    def forward(self, x):
        x = self.net(x)
        return x


def make_nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU(),
    )


if __name__ == '__main__':
    model = NiN(10).cuda()                          # 实例化模型，并放置在 GPU 上
    x = torch.rand(size=(1, 1, 224, 224)).cuda()    # 创建输入张量，并放置在 GPU 上
    output = model(x)
    for layer in model.net:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape:\t', x.shape)
    print(output.size())
