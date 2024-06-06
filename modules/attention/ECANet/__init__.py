import torch
import torch.nn as nn
from math import log

from utils.logger import log_model_params


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
    仅需 k=3（默认设置时）个参数（与通道数的 log2 线性相关），就能得到于 SENet（参数与通道数相关） 相比 0.72%(Top-1) / 0.27%(Top-5) 的提升
    """
    def __init__(self, in_channels, b=1, gamma=2):
        """

        Args:
            in_channels: 输入通道数
            b: 与 gamma 共同决定 k 的大小
            gamma: 与 gamma 共同决定 k 的大小
        """
        super(ECABlock, self).__init__()
        self.gamma = gamma

        t = int(abs(log(in_channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                   # [N, C, H, W] -> [N, C, 1]
        y = (y.squeeze(-1)                     # [N, C, 1] -> [N, C]
              .transpose(-1, -2))              # [N, C] -> [C, N]
        y = self.conv(y)                       # [C, N]
        y = (y.transpose(-1, -2)               # [C, N] -> [N, C]
              .unsqueeze(-1))                  # [N, C] -> [N, C, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)              # [N, C, H, W]


if __name__ == '__main__':
    x_input = torch.randn(2, 16, 32, 32)
    eca_block = ECABlock(16)

    log_model_params(eca_block, x_input.shape)