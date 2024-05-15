import os

import numpy as np
import torch
from PIL import Image


def save_label_as_png(mask, device, color_map, dir_path='./logs/labels', file_name=None):
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

    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        finally:
            pass

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
    masks_rgb = masks_rgb.permute(0, 3, 1, 2)           # [8, 256, 256, 3] -> [8, 3, 256, 256]
    masks_rgb = torch.chunk(masks_rgb, masks_rgb.shape[0], dim=0)
    for j, image in enumerate(masks_rgb):
        image = image.squeeze(dim=0)                    # [1, 3, 256, 256] -> [3, 256, 256]
        image = image.permute(1, 2, 0)                  # [3, 256, 256] -> [256, 256, 3]
        print("unique:", torch.flatten(image, start_dim=0, end_dim=1).unique(dim=0).squeeze())
        image = Image.fromarray(image.cpu().numpy().astype(np.uint8), mode='RGB')
        image_path = os.path.join(dir_path, f"{file_name}_{j}")
        image.save(image_path, format='PNG')


def save_mask_as_png(mask, dir_path='./logs/masks', file_name=None):
    """
    将 mask 保存为图像文件，比如 [3, 256, 256] -> [256, 256, 3] 然后储存
    Args:
        mask:
        dir_path:
        file_name:

    Returns:

    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except:
            pass
    k = 0

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)

    mask = mask.permute(1, 2, 0)                  # [3, 256, 256] -> [256, 256, 3]
    mask = mask * 255.0                           # 0 ~ 1 -> 0 ~ 255
    image = Image.fromarray(mask.cpu().numpy().astype(np.uint8), mode='RGB')
    while True:
        save_file_name = f"{file_name}{k}.png" if file_name is not None else f"{k}.png"
        image_path = os.path.join(dir_path, save_file_name)
        if not os.path.exists(image_path):
            break
        k += 1
    image.save(image_path, format='PNG')


