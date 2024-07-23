import torch

from utils.log.info import print_, warning


def assert_on_cuda():
    """
    检查是否部署在 CUDA 设备上，如果不是，则发出警告，并退出程序
    """
    # 部署 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_("Using device: ", device)
    # 如果设备是 GPU，则打印 CUDA 版本
    if device.type.lower() == "cuda":
        print_("CUDA version: ", torch.version.cuda)
    # 如果设备是 CPU，则发出警告，并退出程序
    if device.type.lower() == "cpu":
        warning("Using CPU for training. This may be slow.")
        print_("torch version: ", torch.__version__)
        print_("自行到网站：https://pytorch.org 下载对应版本 cuda")
        exit()

    return device
