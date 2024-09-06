from typing import Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from modeling.utils import switch_HWC
from modules.mlp import MLP


# 缩放模块
class ScalableModule(nn.Module):
    """
    A module that can be scaled by a learnable parameter.
    如果输入为 [B, N, C]，则对 N 维度进行缩放，输出为 [B, N, C]
    """
    init_values: Union[float, torch.Tensor] = 1e-5

    def __init__(
        self, *,
        block: nn.Module,
        dim: int,
        inplace: bool = False,
    ):
        """
        Args:
            block:
            dim: 相当于 [B, N, C] 中的 C
            inplace:
        """
        super().__init__()
        self.block = block
        self.gamma = nn.Parameter(self.init_values * torch.ones(dim))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)

        if isinstance(x, tuple):    # 如果 block 返回多个值，则只取第一个值
            x = x[0]
        x = x.mul_(self.gamma) if self.inplace else x * self.gamma  # [B, N, C] * [C] -> [B, N, C]
        return x


# encoder 中的模块
# 一个注意力模块，参考 Transformer 中 Encoder 的模块
class SelfAttentionBlock(nn.Module):
    def __init__(
        self, *,
        dim: int,
        num_heads: int = 8,
        attn: nn.Module,
        ffn: nn.Module,
        drop_path_rate: float = 0.0,
        norm_layer: [nn.Module, (nn.Module, nn.Module)] = nn.LayerNorm,
        **kwargs
    ):
        """
        Args:
            dim: 输入的维度
            num_heads: 注意力头数
            attn: 注意力模块
            ffn: 前馈网络模块
            drop_path_rate: dropout 率
            norm_layer: 归一化层，可以是 nn.Module 或 (nn.Module, nn.Module)
            kwargs: 包括 'q_stride' 和 'dim_out'，分别表示 Q pooling 的步长和输出维度
        """
        super().__init__()
        if not isinstance(norm_layer, tuple):
            norm_layer_1 = norm_layer_2 = norm_layer
        elif isinstance(norm_layer, tuple) and len(norm_layer) == 2:
            norm_layer_1, norm_layer_2 = norm_layer
        else:
            raise ValueError(f"norm_layer should be nn.Module or (nn.Module, nn.Module), instead of {type(norm_layer)}")

        self.dim = dim
        self.num_heads = num_heads

        # Attention + Q pooling
        self.norm1 = norm_layer_1(dim)
        self.attn = attn
        if 'q_stride' in kwargs:
            self.q_stride = kwargs['q_stride']
            self.pool = nn.MaxPool2d(kernel_size=self.q_stride, stride=self.q_stride, ceil_mode=False)
        else:
            self.q_stride = None

        # MLP
        self.norm2 = norm_layer_2(dim)
        self.mlp = ffn

        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        if 'dim_out' in kwargs:
            self.dim_out = kwargs['dim_out']
            if self.dim != self.dim_out:
                self.proj = nn.Linear(dim, self.dim_out)
        else:
            self.dim_out = dim

    def forward(self, x: torch.Tensor, return_attention=False) -> torch.Tensor:
        def drop(_x: torch.Tensor, _x_attn: torch.Tensor):
            if self.training and self.drop_path_rate > 0.0:
                _x = _x + self.drop_path(_x_attn)
            else:
                _x = _x + _x_attn

            return _x

        # Attention + Q pooling, norm -> (Q pooling) -> attention -> drop -> add
        x_norm = self.norm1(x)
        if self.q_stride and self.dim != self.dim_out:  # Q pooling
            x = switch_HWC(self.proj(x_norm), self.pool)
        x_attn = self.attn(x_norm)
        if return_attention:
            return x_attn
        x = drop(x, x_attn)

        # MLP, norm -> mlp -> drop -> add
        x_norm = self.norm2(x)
        x = drop(x, self.mlp(x_norm))

        return x


if __name__ == '__main__':
    from modules.attention import MultiheadAttention

    # Test SelfAttentionBlock
    x_input = torch.randn(1, 128, 32, 32)
    _block = SelfAttentionBlock(
        dim=128,
        dim_out=256,
        num_heads=8,
        q_stride=8,
        attn=MultiheadAttention(d_model=128, num_heads=8),
        ffn=MLP(input_dim=128, hidden_dim=256, output_dim=128, num_layers=3),
        drop_path_rate=0.1,
    )

    y_output = _block(x_input)
    print(y_output.shape)
