import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return x * torch.sigmoid(x)


class ReLUSquared(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return F.relu(x) ** 2


class LaplacianAttentionFunction(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5
