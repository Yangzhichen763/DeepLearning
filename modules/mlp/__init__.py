from typing import Optional, Callable

import torch.nn as nn
import torch.nn.functional as F


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
            drop_out: float = 0.0,
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
        self.act = activation()  # 每层（除了最后一层）都有的激活函数

        self.drop = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.drop(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

