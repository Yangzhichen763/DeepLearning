import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from modeling.utils import switch_HWC


# 参考代码：https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers

        hidden_layers = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden_layers, hidden_layers + [output_dim])
        )

        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self, *,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_stride: int,
        attn,
        mlp,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_stride = q_stride

        # Attention + Q pooling
        self.norm1 = norm_layer(dim)
        self.attn = attn
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        # MLP
        self.norm2 = norm_layer(dim)
        self.mlp = mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
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
    block = SelfAttentionBlock(
        dim=128,
        dim_out=256,
        num_heads=8,
        q_stride=8,
        attn=MultiheadAttention(d_model=128, num_heads=8),
        mlp=MLP(input_dim=128, hidden_dim=256, output_dim=128, num_layers=3),
        drop_path=0.1,
    )

    y_output = block(x_input)
    print(y_output.shape)