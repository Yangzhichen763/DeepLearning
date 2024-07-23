import torch
import torch.nn as nn
from utils.torch.nn import flatten_     # flatten_ 相当于 view，但是更加灵活

from utils.log.model import log_model_params


"""
Criss-Cross Attention Network (CCNet)
论文链接 2018-2020：https://arxiv.org/abs/1811.11721
"""


class CrissCrossBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(CrissCrossBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)

    # noinspection PyTestUnpassedFixture
    def forward(self, x):
        """
        Returns:
            x: [B, C/r, H, W]
            h: [B*W, C/r, H]
            w: [B*H, C/r, W]
        """
        # [B, C, H, W] -> [B, C/r, H, W]
        x = self.conv(x)
        # [B, C/r, H, W] -> [B, W, C/r, H] -> [B*W, C/r, H]
        h = x.permute(0, 3, 1, 2).contiguous().flatten_(start_dim=-4, end_dim=-3)
        # [B, C/r, H, W] -> [B, H, C/r, W] -> [B*H, C/r, W]
        w = x.permute(0, 2, 1, 3).contiguous().flatten_(start_dim=-4, end_dim=-3)
        return x, h, w  # [B, C/r, H, W], [B*W, C/r, H], [B*H, C/r, W]


class CrissCrossAttention(nn.Module):
    """
    Criss-Cross Attention Block 十字交叉注意力模块
    """
    def __init__(self, in_channels, reduction=8):
        super(CrissCrossAttention, self).__init__()
        self.q = CrissCrossBlock(in_channels, reduction)
        self.k = CrissCrossBlock(in_channels, reduction)
        self.v = CrissCrossBlock(in_channels, reduction=1)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape

        # [B, C/r, H, W], [B*W, C/r, H], [B*H, C/r, W]
        q, q_h, q_w = self.q(x)
        k, k_h, k_w = self.k(x)
        v, v_h, v_w = self.v(x)

        energy_h = torch.bmm(q_h.transpose(-1, -2), k_h)   # [B*W, H, C/r] x [B*H, C/r, H] -> [B*W, H, H]
        energy_w = torch.bmm(q_w.transpose(-1, -2), k_w)   # [B*H, W, C/r] x [B*W, C/r, W] -> [B*H, W, W]
        energy_concat = torch.cat(
            tensors=[
                energy_h.unflatten(dim=-3, sizes=(-1, w))   # [B*W, H, H] -> [B, W, H, H]
                        .transpose(-2, -3),                 # [B, W, H, H] -> [B, H, W, H]
                energy_w.unflatten(dim=-3, sizes=(-1, h))   # [B*H, W, W] -> [B, H, W, W]
            ], dim=-1)  # [B, H, W, H+W]
        attn: torch.Tensor = self.softmax(energy_concat)

        attn_h = (attn[..., 0:h]                            # [B, H, W, H+W] -> [B, H, W, H]
                  .transpose(-2, -3).contiguous()           # [B, H, W, H] -> [B, W, H, H]
                  .view_as(energy_h))                       # [B, W, H, H] -> [B*W, H, H]
        attn_w = (attn[..., h:h+w]                          # [B, H, W, H+W] -> [B, H, W, W]
                  .transpose(-2, -3).contiguous()           # [B, H, W, W] -> [B, W, H, W]
                  .view_as(energy_w))                       # [B, W, H, W] -> [B*H, W, W]

        out_h = torch.bmm(v_h, attn_h.transpose(-1, -2))    # [B*W, C, H] x [B*W, H, H] -> [B*W, C, H]
        out_w = torch.bmm(v_w, attn_w.transpose(-1, -2))    # [B*H, C, W] x [B*H, W, W] -> [B*H, C, W]

        out = (out_h.unflatten(dim=-3, sizes=(-1, w))       # [B*W, C, H] -> [B, W, C, H]
                    .permute(0, 2, 3, 1)                    # [B, W, C, H] -> [B, C, H, W]
               +
               out_w.unflatten(dim=-3, sizes=(-1, h))       # [B*H, C, W] -> [B, H, C, W]
                    .permute(0, 2, 1, 3))                   # [B, H, C, W] -> [B, C, H, W]
        return self.gamma * out + x                         # [B, C, H, W]


if __name__ == '__main__':
    x_input = torch.randn(2, 64, 16, 16)
    cc_attn = CrissCrossAttention(in_channels=x_input.shape[1])

    log_model_params(cc_attn, input_data=x_input)


