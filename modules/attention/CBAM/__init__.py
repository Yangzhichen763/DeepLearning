import torch
import torch.nn as nn


"""
CBAM: Convolutional Block Attention Module
CBAM 是一种混合注意力机制，是对空间注意力机制 SAM(Spatial Attention Module) 和通道注意力机制 CAM(Channel Attention Module) 的组合。
论文链接 2018：https://arxiv.org/abs/1807.06521（已阅读x1）
"""


class ChannelAttention(nn.Module):
    """
    相比于 SENet 更加灵活，有 ratio 可以控制通道方向上的压缩比率。
    """
    def __init__(self, in_channels, ratio=16):
        """
        Args:
            in_channels: 输入通道数
            ratio: 通道方向压缩比率，默认值为 16。
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        y = self.sigmoid(avg_out + max_out)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """

        Args:
            kernel_size: kernel_size=7 效果最好，一般取值 3 或 7。
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 论文中尝试过使用将全局平均池化和全局最大池化替换为 1x1 卷积，但是效果没有前者好，Top-1 和 Top-5 误差差距在 0.3% 左右。
        avg_out = torch.mean(x, dim=1, keepdim=True)        # [N, C, H, W] -> [N, 1, H, W] 在 C 方向求平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # [N, C, H, W] -> [N, 1, H, W] 在 C 方向求最大值
        y = torch.cat([avg_out, max_out], dim=1)     # 2 × [N, 1, H, W] -cat-> [N, 2, H, W]
        y = self.conv(y)                                    # [N, 2, H, W] -> [N, 1, H, W]
        y = self.sigmoid(y)
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 论文中试过 CA 和 SA 的先后顺序，并行的也试过，最终得到先 CA 再 SA 的效果最好。
        self.ca = ChannelAttention(out_channels)    # 相对于普通的 ResNet，添加这行代码(CA 模块)
        self.sa = SpatialAttention()                # 相对于普通的 ResNet，添加这行代码(SA 模块)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.feature_layer(x)
        out *= self.ca(out)     # 相对于普通的 ResNet，添加这行代码(CA 模块)
        out *= self.sa(out)     # 相对于普通的 ResNet，添加这行代码(SA 模块)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


if __name__ == '__main__':
    from utils.log.model import log_model_params

    x_input = torch.randn(2, 3, 224, 224)
    model = BasicBlock(3, 64, 2)

    log_model_params(model, input_data=x_input)
