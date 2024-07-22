from utils.torch.deploy import *
from utils.torch.classify import *
from utils.torch.dataset.datapicker import *

__all__ = ["pick_sequential", "pick_random",
           "assert_on_cuda",
           "classify", "deploy", "load", "save",
           "extract", "expend_as", "expend"]


def extract(a, t, x_shape):
    """
    将 a 按照 t 进行切片，并将切片后的结果展平为 shape=[batch_size, 1, 1, 1, ...]
    Args:
        a (torch.Tensor):
        t (torch.Tensor | int):
        x_shape (tuple|torch.Size):
    """
    if isinstance(t, torch.Tensor):
        out = a.gather(dim=-1, index=t)
        return out.view(x_shape[0], *((1,) * (len(x_shape) - 1)))
    elif isinstance(t, int):
        out = a[t]
        return out.repeat(x_shape[0], *((1,) * (len(x_shape) - 1)))
    else:
        raise ValueError("t must be int or tensor")


def expend_as(a, x_shape):
    """
    将 a 拓展为 shape=[batch_size, 1, 1, 1, ...]
    Args:
        a (torch.Tensor):
        x_shape (tuple|torch.Size):
    """
    assert a.shape[0] == 1, "a must have batch_size=1"
    return a.repeat(x_shape[0], *((1,) * (len(x_shape) - 1)))


def expend(a):
    """
    将 a 拓展为 shape=[batch_size, 1, 1, 1]
    """
    return a[:, None, None, None]
