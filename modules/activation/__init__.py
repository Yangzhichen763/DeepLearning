import torch
import torch.nn as nn


class Swish(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return x * torch.sigmoid(x)
