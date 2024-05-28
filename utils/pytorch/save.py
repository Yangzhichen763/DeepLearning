import os
from utils.os import get_unique_full_path

import numpy as np
from torchvision import transforms

import torch


def as_pt(model, save_path=None, file_name=None):
    """
    保存模型为 PyTorch 格式.
    Args:
        model (torch.nn.Module): 要保存的 PyTorch 模型.
        save_path (str): 模型保存路径，包含文件名和后缀名 .pt.
        file_name (str): 模型保存文件名，不包含后缀名 .pt. 只有在 save_path 为 None 时，该参数才生效.
    """
    if save_path is None:
        if file_name is None:
            save_path = "./models/untitled.pt"
        else:
            save_path = f"./models/{file_name}.pt"

    print("Saving model to: ", save_path)
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), save_path)
    print("Model saved.")


def as_onnx(model, dummy_input, save_path=None, file_name=None, input_names=None, output_names=None):
    """
    保存模型为 ONNX 格式.
    Args:
        model (torch.nn.Module): 要保存的 PyTorch 模型.
        dummy_input (torch.Tensor): 模型的输入，用于导出 ONNX 模型.
        save_path (str): 模型保存路径，包含文件名和后缀名 .onnx.
        file_name (str): 模型保存文件名，不包含后缀名 .onnx. 只有在 save_path 为 None 时，该参数才生效.
        input_names (list): 模型输入的名称.
        output_names (list): 模型输出的名称.
    """
    if save_path is None:
        if file_name is None:
            save_path = "./models/untitled.pt"
        else:
            save_path = f"./models/{file_name}.pt"

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.onnx.export(model,
                      dummy_input,
                      save_path,
                      input_names=input_names,
                      output_names=output_names)


def tensor_to_image(tensor, save_path=None, file_name=None, epoch=None):
    """
    将 PyTorch 张量保存为图像.

    Args:
        tensor (torch.Tensor): 要保存的 PyTorch 张量. 形状为 [N, C, H, W] 或 [C, H, W].
        save_path (str): 图像保存路径，包含文件名和后缀名 .png.
        file_name (str): 图像保存文件名，不包含后缀名 .png. 只有在 save_path 为 None 时，该参数才生效.
        epoch (int): 训练轮数，可选参数，默认为 None.
    """
    if save_path is None:
        if file_name is None:
            save_path = "./logs/datas/untitled.png"
        else:
            save_path = f"./logs/datas/{file_name}.png"

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    tensor = tensor.detach()
    if tensor.dim() not in [2, 3, 4]:
        raise ValueError("The tensor should be 2, 3 or 4 dimensions. "
                         f"instead of shape={tensor.shape}.")
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)        # [H, W] -> [1, H, W]
    if tensor.dim() != 4:
        tensor = tensor.unsqueeze(0)        # [C, H, W] -> [1, C, H, W]
    batch_size = tensor.shape[0]

    print(1, torch.unique(tensor))
    tensor = (tensor
              .permute(0, 2, 3, 1)          # [N, C, H, W] -> [N, H, W, C]
              .cpu().numpy())

    to_pil = transforms.ToPILImage()
    for i in range(tensor.shape[0]):
        img = to_pil(tensor[i])             # [H, W, C]
        print(2, torch.unique(transforms.ToTensor()(img)))
        if epoch is not None:
            image_save_path = save_path.replace(".png", f"_{epoch * batch_size + i}.png")
        else:
            image_save_path = get_unique_full_path(save_path)
        img.save(image_save_path)

