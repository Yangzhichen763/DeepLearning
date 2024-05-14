import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    """
    DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """
    MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class DownBilinear(nn.Module):
    """
    DoubleConv -> DownSample
    """
    def __init__(self, in_channels, out_channels):
        super(DownBilinear, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        residual = self.conv(x)
        x = self.down(residual)
        return x, residual


class Up(nn.Module):
    """
    UpSample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UpBilinear(nn.Module):
    """
    UpSample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        """

        Args:
            in_channels: 仅仅只有输入的通道数，没有算上 concat 后的通道数
            out_channels:
        """
        super(UpBilinear, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(2 * in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # [N, C, H, W]
        delta_h = x2.shape[-2] - x1.shape[-2]
        delta_w = x2.shape[-1] - x1.shape[-1]
        # 将 x1 填充到 x2 的尺寸
        x1 = F.pad(x1, [delta_w // 2, delta_w - delta_w // 2,
                        delta_h // 2, delta_h - delta_h // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """
    Conv
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

