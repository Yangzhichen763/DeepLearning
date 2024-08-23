import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import DropPath


"""
ConvNeXt
论文链接：https://arxiv.org/abs/2201.03545
"""


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CXBlock(nn.Module):
    """
    参考代码：https://github.com/facebookresearch/ConvNeXt
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    """
    def __init__(
            self,
            dim,
            kernel_size=7,
            padding=3,
            drop_path=0.0,
            layer_scale_init_value=1e-6,
            use_dwconv=True
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(                # depthwise conv
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim if use_dwconv else 1
        )
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise conv, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)   # [B, C, H, W] -> [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)   # [B, H, W, C] -> [B, C, H, W]

        x = shortcut + self.drop_path(x)
        return x
