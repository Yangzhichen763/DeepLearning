import torch
import torch.nn as nn


"""
GLU: Gated Linear Unit
论文地址 2020：https://arxiv.org/abs/2002.05202
"""


class GLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.in_fc = nn.Linear(in_features, hidden_features * 2)
        self.act = act_layer()
        self.out_fc = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # noinspection PyPep8Naming
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, N, C]
        """
        x, v = self.in_fc(x).chunk(2, dim=-1)  # [B, N, Cin] -> 2*[B, N, dim]
        x = self.act(x) * v
        x = self.drop(x)
        x = self.out_fc(x)                     # [B, N, dim] -> [B, N, Cout]
        x = self.drop(x)
        return x
