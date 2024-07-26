import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log


"""
ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
ECA-Net 是通道注意力机制 CAM(Channel Attention Module)
论文链接 2019-2020：https://arxiv.org/abs/1910.03151   

论文细节：
1. SENet中的降维会给通道注意力机制带来副作用，并且捕获所有通道之间的依存关系是效率不高的且是不必要的。
2. ECA注意力机制模块直接在全局平均池化层之后使用1x1卷积层，去除了全连接层。
   该模块避免了维度缩减，并有效捕获了跨通道交互。并且ECA只涉及少数参数就能达到很好的效果。
3. ECANet通过一维卷积 Conv1D 来完成跨通道间的信息交互，卷积核的大小通过一个函数来自适应变化，使得通道数较大的层可以更多地进行跨通道交互。
"""


class ECABlock(nn.Module):
    """
    ECA-Net block
    \n仅需 k=3（默认设置时）个参数（与通道数的 log2 线性相关），就能得到于 SENet（参数与通道数相关） 相比 0.72%(Top-1) / 0.27%(Top-5) 的提升
    """
    def __init__(self, in_channels, kernel_size=3):
        """

        Args:
            in_channels: 输入通道数
            kernel_size: 卷积核大小
        """
        super(ECABlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1,
                              kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                   # [N, C, H, W] -> [N, C, 1, 1]
        y = (y.squeeze(-1)                     # [N, C, 1, 1] -> [N, C, 1]
              .transpose(-1, -2))              # [N, C, 1] -> [N, 1, C]
        y = self.conv(y)                       # [N, 1, C] -> [N, 1, C]
        y = (y.transpose(-1, -2)               # [N, 1, C] -> [N, C, 1]
              .unsqueeze(-1))                  # [N, C, 1] -> [N, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)              # [N, C, 1, 1] -> [N, C, H, W]


class ECABlock_NS(nn.Module):
    """
    ECA-NS block
    \n具体见论文 2019-2020：https://arxiv.org/abs/1910.03151
    \n与 ECA-Net 相比，ECA-NS 增加了分组卷积，每个 channel 使用 k 个参数进行 channel 之间的交互，这样可以避免不同 Group 之间的信息隔离问题
    \n参数量为 k × C，准确率略逊于 ECA-Net
    """
    def __init__(self, in_channels, kernel_size=3):
        super(ECABlock_NS, self).__init__()
        self.kernel_size = kernel_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels, in_channels,
                              kernel_size=kernel_size, bias=False, groups=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                   # [N, C, H, W] -> [N, C, 1, 1]
        y = y.transpose(-1, -3)                # [N, C, 1, 1] -> [N, 1, 1, C]
        y = F.unfold(y,                        # [N, 1, 1, C] -> [N, k, C]
                     kernel_size=(1, self.kernel_size),
                     padding=(0, (self.kernel_size - 1) // 2)
                     ).transpose(-1, -2)       # [N, k, C] -> [N, C, k]
        y = self.conv(y)                       # [N, C, k] -> [N, C, 1]
        y = y.unsqueeze(-1)                    # [N, C, 1] -> [N, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)              # [N, C, 1, 1] -> [N, C, H, W]


def eca_layer(in_channels, b=1, gamma=2):
    """
    ECA-Net layer
    Args:
        in_channels: 输入通道数
        b: 与 gamma 共同决定 k 的大小
        gamma: 与 gamma 共同决定 k 的大小
    """
    t = int(abs(log(in_channels, 2) + b) / gamma)
    kernel_size = t if t % 2 else t + 1
    return ECABlock(in_channels, kernel_size)


if __name__ == '__main__':
    from utils.log import log_model_params

    x_input = torch.randn(2, 16, 32, 32)
    eca_block = ECABlock(16)

    log_model_params(eca_block, input_data=x_input)