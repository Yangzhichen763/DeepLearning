import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_as(a: torch.Tensor, b: torch.Tensor):
    """
    [N, B] -[N, B, C]-> [N, B, 1]
    """
    assert b.dim() >= a.dim(), f"b should have at least as many dimensions as a, instead of {b.dim()} and {a.dim()}"
    dim_delta = b.dim() - a.dim()
    return a.view(*a.shape, *([1] * dim_delta))


def extract_dim(a: torch.Tensor, dim: int):
    """
    [N, B] -dim=-2-> [N, 1, B]
    """
    shape = list(a.shape)
    shape.insert((dim + a.dim() + 1) % (a.dim() + 1), 1)
    return a.view(*shape)


def squeeze_as(a: torch.Tensor, b: torch.Tensor):
    """
    [N, B, 1] -[N, B]-> [N, B]
    """
    assert 1 in a.shape, f"a should have at least one dimension of size 1, instead of {a.shape}"
    dim_delta = a.dim() - b.dim()
    return a.squeeze(dim=list(range(-dim_delta, 0)))


class MaskedLinear(nn.Module):
    """
    相当于 Drop Connect
    """
    def __init__(self, input_dim, output_dim, bias=True, **factory_kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty((output_dim, input_dim), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, weight_mask=None, bias_mask=None):
        weight: torch.Tensor = self.weight
        if weight_mask is not None:
            weight = weight.masked_fill(weight_mask == 0, 0)

        bias: torch.Tensor = self.bias
        if bias_mask is not None:
            bias = bias.masked_fill(bias_mask == 0, 0)

        return F.linear(x, weight, bias)


class AttentionLinear(nn.Module):
    """
    相当于 Soft Drop Connect
    """
    def __init__(self, input_dim, output_dim, bias=True, **factory_kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty((output_dim, input_dim), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, attention):
        """
        Args:
            x: [N, input_dim]
            attention: [N, input_dim, output_dim], attention 的维度要和 weight 的维度一致
        """
        weight: torch.Tensor = self.weight
        assert weight.shape[-2:] == attention.shape[-2:], \
            (f"The attention shape should be the same as the weight shape. "
             f"instead of {attention.shape[-2:]} and {weight.shape[-2:]}")

        weight = (weight * attention).transpose(-1, -2)
        weighted_x = extract_dim(x, dim=-2) @ weight
        weighted_x = torch.squeeze(weighted_x, dim=-2)
        return weighted_x + self.bias


if __name__ == '__main__':
    x = torch.randn(3, 28)
    attention = torch.randn(3, 29, 28)
    attention_linear = AttentionLinear(28, 29)
    print(attention_linear(x, attention).shape)
