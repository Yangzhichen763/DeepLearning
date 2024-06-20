import torch


def linear(start, end, steps, **kwargs):
    return torch.linspace(
        start=start,
        end=end,
        steps=steps,
        dtype=torch.float32,
        **kwargs)


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

