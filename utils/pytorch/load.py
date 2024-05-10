import torch


def from_model(model, load_path, device):
    """
    加载模型.
    Args:
        model (torch.nn.Module): 要加载的 PyTorch 模型.
        load_path (str): 模型加载路径.
        device (str): 加载设备, 比如：'cpu' 或 'cuda'.
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
