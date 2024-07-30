import torch
import torch.nn as nn
import torch.nn.functional as F
from morphology import MorphOpen2d, MorphClose2d


"""
  - 形态编码器提取域不变性特征，确保了生成样本的可靠性
  - 空间-高光谱信息联合的语义编码器，确保了生成样本的有效性
论文链接 2022：https://arxiv.org/abs/2209.01634
参考文章：https://zhuanlan.zhihu.com/p/562084403
参考代码：https://github.com/YuxiangZhang-BIT/IEEE_TIP_SDEnet/blob/main/network/discriminator.py
"""


class MorphNet(nn.Module):
    def __init__(self, in_channels):
        super(MorphNet, self).__init__()
        morph_channels = 1
        morph_kernel_size = 3

        self.conv = nn.Conv2d(in_channels, morph_channels, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(2)
        self.morph_open = MorphOpen2d(morph_channels, morph_channels, morph_kernel_size, soft_max=False)
        self.morph_close = MorphClose2d(morph_channels, morph_channels, morph_kernel_size, soft_max=False)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x_open = self.morph_open(x)
        x_close = self.morph_close(x)
        x_top_hat = x - x_open
        x_black_hat = x_close - x
        x_morph = torch.cat((x_top_hat, x_black_hat, x_open, x_close), 1)

        return x_morph
