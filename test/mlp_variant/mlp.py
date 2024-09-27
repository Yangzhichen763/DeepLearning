import torch
import torch.nn as nn
import torch.nn.functional as F


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
        weight = weight * attention
        return F.linear(x, weight, self.bias)

