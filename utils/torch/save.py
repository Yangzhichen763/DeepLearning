import os

import torchvision
import PIL.Image as Image

from utils.os import get_unique_full_path, insert_before_dot
from tqdm import tqdm

import numpy as np

import torch


def as_pt(model, save_path=None, file_name=None, path_unique=False):
    """
    保存模型为 PyTorch 格式.
    Args:
        model (torch.nn.Module): 要保存的 PyTorch 模型.
        save_path (str): 模型保存路径，包含文件名和后缀名 .pt.
        file_name (str): 模型保存文件名，不包含后缀名 .pt. 只有在 save_path 为 None 时，该参数才生效.
        path_unique (bool): 是否为保存路径生成唯一路径.
    """
    if save_path is None:
        if file_name is None:
            save_path = "./models/untitled.pt"
        else:
            if file_name[-3:] == ".pt":
                save_path = f"./models/{file_name}"
            else:
                save_path = f"./models/{file_name}.pt"

    if path_unique:
        save_path = get_unique_full_path(save_path)
    tqdm.write(f"Saving model to: {save_path}")
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), save_path)


def as_onnx(model, dummy_input, save_path=None, file_name=None,
            input_names=None, output_names=None, path_unique=False):
    """
    保存模型为 ONNX 格式.
    Args:
        model (torch.nn.Module): 要保存的 PyTorch 模型.
        dummy_input (torch.Tensor): 模型的输入，用于导出 ONNX 模型.
        save_path (str): 模型保存路径，包含文件名和后缀名 .onnx.
        file_name (str): 模型保存文件名，不包含后缀名 .onnx. 只有在 save_path 为 None 时，该参数才生效.
        input_names (list): 模型输入的名称.
        output_names (list): 模型输出的名称.
        path_unique (bool): 是否为保存路径生成唯一路径.
    """
    if save_path is None:
        if file_name is None:
            save_path = "./models/untitled.pt"
        else:
            if file_name[-5:] == ".onnx":
                save_path = f"./models/{file_name}"
            else:
                save_path = f"./models/{file_name}.onnx"

    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    if path_unique:
        save_path = get_unique_full_path(save_path)
    torch.onnx.export(model,
                      dummy_input,
                      save_path,
                      input_names=input_names,
                      output_names=output_names)


def tensor_to_image(tensor, save_path=None, make_grid=False, **kwargs):
    """
    将 PyTorch 张量保存为图像.
    如果传入 file_name，则文件保存路径为以 save_path 为父目录文件名为 file_name.png.
    Args:
        tensor (torch.Tensor): 要保存的 PyTorch 张量. 形状为 [N, C, H, W] 或 [C, H, W].
        save_path (str): 图像保存路径，包含文件名和后缀名 .png.
        make_grid (bool): 是否将 tensor 按照 batch 组合成网格保存为图像.
    """
    if save_path is None:
        if kwargs.get("file_name") is not None:
            file_name: str = kwargs["file_name"]
            save_path = f"./logs/datas/{file_name}.png"
        else:
            save_path = "./logs/datas/untitled.png"
    else:
        if kwargs.get("file_name") is not None:
            file_name: str = kwargs["file_name"]
            save_path = os.path.join(os.path.dirname(save_path), f"{file_name}.png")

    # 寻炸或创建图像保存目录
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    # 将图像转成 PIL 格式
    if tensor.dim() not in [2, 3, 4]:
        raise ValueError("The tensor should be 2, 3 or 4 dimensions. "
                         f"instead of shape={tensor.shape}.")
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)        # [H, W] -> [1, H, W]
    if tensor.dim() != 4:
        tensor = tensor.unsqueeze(0)        # [C, H, W] -> [1, C, H, W]
    batch_size = tensor.shape[0]
    # tensor = (tensor.permute(0, 2, 3, 1))   # [N, C, H, W] -> [N, H, W, C]
    if tensor.dtype == torch.float32 and not (tensor > 1).any():
        tensor = (255 * tensor)             # [0, 1] -> [0, 255]
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(torch.uint8)     # -> uint8

    to_pil = torchvision.transforms.ToPILImage()
    if make_grid:
        # 将 tensor 按照 batch 组合成网格保存为图像
        grid = torchvision.utils.make_grid(tensor, **kwargs)
        image = to_pil(grid)
        if kwargs.get("batch") is not None:
            batch: int = kwargs["batch"]
            image_save_path = insert_before_dot(save_path, f"_{batch}")
        else:
            image_save_path = get_unique_full_path(save_path)
        image.save(image_save_path)
    else:
        # 将 tensor 按照 batch 分批保存为图像
        for (i, tensor_i) in enumerate(tensor):
            image = to_pil(tensor_i)
            if kwargs.get("batch") is not None:
                batch: int = kwargs["batch"]
                image_save_path = insert_before_dot(save_path, f"_{batch * batch_size + i}")
            else:
                image_save_path = get_unique_full_path(save_path)
            image.save(image_save_path)


def to_image(data, save_path=None, make_grid=False, **kwargs):
    if isinstance(data, torch.Tensor):
        tensor_to_image(data, save_path=save_path, make_grid=make_grid, **kwargs)
    elif isinstance(data, np.ndarray):
        tensor_to_image(torch.from_numpy(data), save_path=save_path, make_grid=make_grid, **kwargs)


if __name__ == '__main__':
    _tensor = torch.randn(2, 3, 256, 256).clamp(0, 1).round()
    print(_tensor.type(), _tensor.unique())
    tensor_to_image(_tensor, save_path="E:/Developments/PythonLearning/DeepLearning/test/logs/datas/test.png")