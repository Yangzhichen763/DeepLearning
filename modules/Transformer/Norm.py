import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / torch.sqrt(std + self.eps)
        out = self.gamma * out + self.beta
        return out
