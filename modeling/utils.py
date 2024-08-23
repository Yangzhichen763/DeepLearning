import torch
import torch.nn as nn


# noinspection PyPep8Naming
def switch_HWC(x, module, **kwargs):
    """
    Args:
        x (torch.Tensor): input tensor, [B, H, W, C]
        module (nn.Module): module to be applied
    """
    x = x.permute(0, 3, 1, 2)   # [B, H, W, C] -> [B, C, H, W]
    x = module(x, **kwargs)
    x = x.permute(0, 2, 3, 1)   # [B, C, H, W] -> [B, H, W, C]
    return x


# noinspection PyPep8Naming
def switch_CHW(x, module, **kwargs):
    """
    Args:
        x (torch.Tensor): input tensor, [B, C, H, W]
        module (nn.Module): module to be applied
    """
    x = x.permute(0, 2, 3, 1)   # [B, C, H, W] -> [B, H, W, C]
    x = module(x, **kwargs)
    x = x.permute(0, 3, 1, 2)   # [B, H, W, C] -> [B, C, H, W]
    return x
