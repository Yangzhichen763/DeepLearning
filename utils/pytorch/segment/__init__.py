import os
from utils.pytorch import *

import numpy as np
import torch
from PIL import Image


def save_label_as_png(mask, device, color_map, file_name=None):
    """
    将 mask 按照 colormap 进行颜色映射，并保存为图像文件
    Args:
        mask: mask 中的值是从 0 开始，逐一递增的
        device:
        dir_path:
        file_name:
        color_map: 比如 torch.tensor([[0, 0, 0], [255, 255, 255]]) 为将 0, 1 标签映射为黑色和白色

    Returns:

    """
    if color_map is None:
        raise ValueError("colormap should not be None.")

    colormap = color_map.to(device)

    mask = mask.detach()
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)

    if mask.dim() == 4:
        mask = mask.argmax(dim=1)                       # [8, 21, 256, 256] -> [8, 1, 256, 256]
        output_label = torch.squeeze(mask)              # [8, 1, 256, 256] -> [8, 256, 256]
    elif mask.dim() == 3 or mask.dim() == 2:
        output_label = mask                             # [8, 256, 256] / [256, 256]
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}, should be 4D or 3D or 2D.")

    if output_label.dim() == 2:
        output_label = output_label.unsqueeze(dim=0)    # [256, 256] -> [1, 256, 256]

    # batch_size = mask.shape[0]
    masks_rgb = colormap[output_label]                  # [8, 256, 256] -> [8, 256, 256, 3]
    save.tensor_to_image(masks_rgb, './logs/labels', file_name)


def save_mask_as_png(mask, file_name=None):
    save.tensor_to_image(mask, './logs/masks', file_name)


