import os

import torch


def as_pt(model, save_path):
    """
    保存模型为 PyTorch 格式.
    Args:
        model (torch.nn.Module): 要保存的 PyTorch 模型.
        save_path (str): 模型保存路径，包含文件名和后缀名 .pt.
    """
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), save_path)


def as_onnx(model, dummy_input, save_path, input_names=None, output_names=None):
    """
    保存模型为 ONNX 格式.
    Args:
        model (torch.nn.Module): 要保存的 PyTorch 模型.
        dummy_input (torch.Tensor): 模型的输入，用于导出 ONNX 模型.
        save_path (str): 模型保存路径，包含文件名和后缀名 .onnx.
        input_names (list): 模型输入的名称.
        output_names (list): 模型输出的名称.
    """
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.onnx.export(model,
                      dummy_input,
                      save_path,
                      input_names=input_names,
                      output_names=output_names)
