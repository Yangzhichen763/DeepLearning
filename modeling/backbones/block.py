from typing import Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from modeling.utils import switch_HWC


# mlp 全连接层
# 参考代码：https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self, *,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_layers: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        self.num_layers = num_layers

        hidden_layers = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden_layers, hidden_layers + [output_dim])
        )

        self.sigmoid_output = sigmoid_output
        self.act = activation()                 # 每层（除了最后一层）都有的激活函数

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


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
        super().__init__()
        self.block = block
        self.gamma = nn.Parameter(self.init_values * torch.ones(dim))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = x.mul_(self.gamma) if self.inplace else x * self.gamma
        return x


# encoder 中的模块
# 一个注意力模块，参考 Transformer 中 Encoder 的模块
class SelfAttentionBlock(nn.Module):
    def __init__(
        self, *,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_stride: Optional[int] = None,
        attn: nn.Module,
        mlp: nn.Module,
        drop_path: float = 0.0,
        norm_layer: [nn.Module, (nn.Module, nn.Module)] = nn.LayerNorm,
    ):
        super().__init__()
        if isinstance(norm_layer, (tuple[nn.Module, nn.Module])):
            norm_layer_1, norm_layer_2 = norm_layer
        elif isinstance(norm_layer, nn.Module):
            norm_layer_1 = norm_layer_2 = norm_layer
        else:
            raise ValueError('norm_layer should be either a Module or a tuple of Modules')

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_stride = q_stride

        # Attention + Q pooling
        self.norm1 = norm_layer_1(dim)
        self.attn = attn
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        # MLP
        self.norm2 = norm_layer_2(dim)
        self.mlp = mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q pooling
        x_norm = self.norm1(x)
        if self.q_stride and self.dim != self.dim_out:
            x = switch_HWC(self.proj(x_norm), self.pool)
        x = x + self.drop_path(self.attn(x_norm))

        # MLP
        x_norm = self.norm2(x)
        x = x + self.drop_path(self.mlp(x_norm))

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
        mlp=MLP(input_dim=128, hidden_dim=256, output_dim=128, num_layers=3),
        drop_path=0.1,
    )

    y_output = _block(x_input)
    print(y_output.shape)
