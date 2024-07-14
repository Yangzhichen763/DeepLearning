import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels[1], kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels[2], kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(in_channels, out_channels[3], kernel_size, stride, padding)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)
