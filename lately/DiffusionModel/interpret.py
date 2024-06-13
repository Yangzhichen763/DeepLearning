import torch


def linear_interpret(start, end, steps, **kwargs):
    return torch.linspace(start, end, steps, dtype=torch.float32, **kwargs)
