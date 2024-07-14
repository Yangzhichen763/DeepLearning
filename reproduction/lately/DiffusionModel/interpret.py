import math

import torch


def linear(start, end, steps, **kwargs):
    return torch.linspace(
        start=start,
        end=end,
        steps=steps,
        dtype=torch.float32,
        **kwargs)


def cosine(start, end, steps, **kwargs):
    """
    原式 â 计算公式为：â(t) = f(t) / f(0)
    原式 f(t) 计算公式为：f(t) = cos(((t / (T + s)) / (1 + s)) · (π / 2))²
    """
    def alpha_hat(t):
        return torch.cos((t + start) / (1 + start) * math.pi / 2) ** 2

    betas = torch.linspace(start=start, end=end, steps=steps + 1, dtype=torch.float32, **kwargs)
    t_curr = betas[:-1]
    t_next = betas[1:]
    betas = torch.min(1 - alpha_hat(t_next) / alpha_hat(t_curr), end)
    return betas


def quad(start, end, steps, **kwargs):
    return torch.linspace(
        start=start ** 0.5,
        end=end ** 0.5,
        steps=steps,
        dtype=torch.float32,
        **kwargs) ** 2


def const(start, end, steps, **kwargs):
    return torch.full(
        size=(steps,),
        fill_value=end,
        dtype=torch.float32,
        **kwargs)


def jsd(start, end, steps, **kwargs):
    """
    1/T, 1/(T-1), 1/(T-2), ..., 1
    """
    return 1.0 / torch.linspace(
        start=steps,
        end=1,
        steps=steps,
        dtype=torch.float32,
        **kwargs)


def sigmoid(start, end, steps, **kwargs):
    def _sigmoid(x):
        return 1 / (torch.exp(-x) + 1)

    betas = torch.linspace(
        start=-6,
        end=6,
        steps=steps,
        dtype=torch.float32,
        **kwargs)
    return _sigmoid(betas) * (end - start) + start


def get(func_name, start, end, steps, **kwargs):
    function = globals().get(func_name)
    if callable(function):
        return function(start, end, steps, **kwargs)
    else:
        raise ValueError(f"Function {func_name} not found in interpret module.")


if __name__ == '__main__':
    print(get('linear', 0, 1, 10))

