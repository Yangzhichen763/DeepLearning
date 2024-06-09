import torch
import torch.nn as nn
from utils.torch.nn import flatten_     # flatten_ 相当于 view，但是更加灵活

from utils.logger.modellogger import log_model_params


"""
Non-Local Neural Networks
论文链接 2018：https://arxiv.org/abs/1711.07971

论文细节：
1. 使用 non-local 对 baseline 结果是有提升的，但是不同相似度计算方法之间差距并不大
2. non-local 加入网络的不同 stage 下性能都有提升,但是对较小的 feature map 提升不大
3. 添加越多的 non-local 模块，效果提升越明显，但是会增大计算量
4. 同时在时域和空域上加入 non-local 操作效果会最好
"""


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.dim_hidden = in_channels // 2

        self.conv_phi = self.make_conv1x1(self.in_channels, self.dim_hidden)
        self.conv_theta = self.make_conv1x1(self.in_channels, self.dim_hidden)
        self.conv_g = self.make_conv1x1(self.in_channels, self.dim_hidden)

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = self.make_conv1x1(self.dim_hidden, self.in_channels)

    @staticmethod
    def make_conv1x1(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    # noinspection PyTestUnpassedFixture
    def forward(self, x: torch.Tensor):
        x_phi = self.conv_phi(x).flatten_(start_dim=-2)     # [N, C, H, W] -> [N, C/2, H, W] -> [N, C/2, H*W]
        x_theta = self.conv_theta(x).flatten_(start_dim=-2)
        x_g = self.conv_g(x).flatten_(start_dim=-2)

        attn = torch.bmm(x_theta.transpose(-1, -2), x_phi)  # [N, H*W, C/2] x [N, C/2, H*W] -> [N, H*W, H*W]
        attn = self.softmax(attn)

        attn_g = torch.bmm(attn, x_g.transpose(-1, -2))     # [N, H*W, H*W] x [N, H*W, C/2] -> [N, H*W, C/2]
        attn_g = (attn_g.transpose(-1, -2)                  # [N. H*W, C/2] -> [N, C/2, H*W]
                  .unflatten(dim=-1, sizes=x.shape[-2:]))   # [N, C/2, H*W] -> [N, C/2, H, W]

        mask = self.conv_mask(attn_g)                       # [N, C/2, H, W] -> [N, C, H, W]
        out = mask + x
        return out


if __name__ == '__main__':
    x_input = torch.randn(3, 16, 64, 64)
    model = NonLocalBlock(in_channels=x_input.shape[1])

    log_model_params(model, input_data=x_input)

