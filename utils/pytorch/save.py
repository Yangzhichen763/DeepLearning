import os
from utils.os import get_unique_full_path
from tqdm import tqdm

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

    tqdm.write(f"Saving model to: {save_path}")
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), save_path)


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


def tensor_to_image(tensor, save_path=None, file_name=None, batch=None):
    """
    将 PyTorch 张量保存为图像.

    Args:
        tensor (torch.Tensor): 要保存的 PyTorch 张量. 形状为 [N, C, H, W] 或 [C, H, W].
        save_path (str): 图像保存路径，包含文件名和后缀名 .png.
        file_name (str): 图像保存文件名，不包含后缀名 .png. 只有在 save_path 为 None 时，该参数才生效.
        batch (int): batch 数，可选参数，默认为 None.
    """
    if save_path is None:
        if file_name is None:
            save_path = "./logs/datas/untitled.png"
        else:
            save_path = f"./logs/datas/{file_name}.png"

    # 寻炸或创建图像保存目录
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 将图像转成 PIL 格式
    tensor = tensor.detach()
    if tensor.dim() not in [2, 3, 4]:
        raise ValueError("The tensor should be 2, 3 or 4 dimensions. "
                         f"instead of shape={tensor.shape}.")
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)        # [H, W] -> [1, H, W]
    if tensor.dim() != 4:
        tensor = tensor.unsqueeze(0)        # [C, H, W] -> [1, C, H, W]
    batch_size = tensor.shape[0]
    tensor = (tensor
              .permute(0, 2, 3, 1))         # [N, C, H, W] -> [N, H, W, C]
    to_pil = transforms.ToPILImage()

    for i in range(tensor.shape[0]):
        if (tensor[i] > 1).any():
            tensor[i] /= 255
        image = to_pil(tensor[i].cpu().numpy().astype(np.float32))     # [H, W, C]
        if batch is not None:
            image_save_path = save_path.replace(".png", f"_{batch * batch_size + i}.png")
        else:
            image_save_path = get_unique_full_path(save_path)
        image.save(image_save_path)


if __name__ == '__main__':
    tensor = torch.randn(2, 3, 256, 256).clamp(0, 1).round()
    print(tensor.unique())
    tensor_to_image(tensor, save_path="E:/Developments/PythonLearning/DeepLearning/lately/UNet/logs/datas2/test.png")