import enum
import math

import einops
import torch
import torch.nn as nn

from ...activation import ReLUSquared, LaplacianAttentionFunction


"""
GAU: Gated Attention Unit
论文链接 2022：https://arxiv.org/abs/2202.10447
参考代码：https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py
"""


# noinspection SpellCheckingInspection
class AttentionFunction(enum.Enum):
    RELUSQUARED = ReLUSquared()
    LAPLACIAN = LaplacianAttentionFunction()


# WTConv 的代码中有一维的 ScaleModule 和 LinearModule 的实现
class LinearModule(nn.Module):
    """
    逐点权重线性模块
    """
    def __init__(self, dims, heads=1):
        super().__init__()
        self.dims = dims
        self.heads = heads

        self.weight = nn.Parameter(torch.ones(heads, dims))
        self.bias = nn.Parameter(torch.zeros(heads, dims))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        out = torch.einsum('... d, h d -> ... h d', x, self.weight) + self.bias
        return out.unbind(dim=-2)


class GAU(nn.Module):
    def __init__(
        self,
        dim: int,
        query_key_dim: int = 128,
        expansion_factor: float = 2,
        dropout: float = 0,
        attention_func=AttentionFunction.RELUSQUARED,
        norm: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm(dim)
        self.attention_func = attention_func.value
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )
        self.offset_scale = LinearModule(query_key_dim, heads=2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    # noinspection PyPep8Naming
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, C]
            mask:
        """
        B, N, C = x.shape

        x_norm = self.norm(x)
        v, gate = self.to_hidden(x_norm).chunk(2, dim=-1)

        qk = self.to_qk(x_norm)
        q, k = self.offset_scale(qk)

        sim = torch.einsum('b i d, b j d -> b i j', q, k)

        attention = self.attention_func(sim / N)
        attention = self.dropout(attention)

        if mask is not None:
            mask = einops.rearrange(mask, 'b j -> b 1 j')
            attention = attention.masked_fill(~mask, 0.)

        out = torch.einsum('b i j, b j d -> b i d', attention, v)
        out = out * gate

        out = self.to_out(out)
        return out