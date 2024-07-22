import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, t_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.t_dim = t_dim

    def forward(self, x):
        half_dim = self.t_dim // 2

        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=x.device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        return embedding


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device=None):
        super(PositionalEmbedding, self).__init__()
        self.embedding = torch.zeros(max_len, d_model, device=device)
        self.embedding.require_grad = False

        pos = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float, device=device)

        self.embedding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.embedding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.embedding[:seq_len, :]
