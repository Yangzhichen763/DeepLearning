import torch

from utils.log.info import print_


def from_model(model, device, load_path=None, file_name=None):
    """
    加载模型.
    Args:
        model (torch.nn.Module): 要加载的 PyTorch 模型.
        load_path (str): 模型加载路径.
        device (str|torch.device): 加载设备, 比如：'cpu' 或 'cuda'.
        file_name (str): 模型文件名，不包含后缀名. 只有在 load_path 为 None 时，该参数才生效.
    """
    if load_path is None:
        if file_name is None:
            raise ValueError("No load_path or file_name is provided! model will not be loaded.")
        else:
            if file_name[-3:] == ".pt":
                load_path = f"./models/{file_name}"
            else:
                load_path = f"./models/{file_name}.pt"

    print_(f"Loading model from {load_path}...", end="")
    model.load_state_dict(torch.load(load_path, map_location=device))
    print_(f"\rModel {load_path} loaded.")
