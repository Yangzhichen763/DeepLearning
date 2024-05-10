import torch
from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device=None):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.require_grad = False

        pos = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout, device=None):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_len, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(x)
        x = self.dropout(token_embedding + positional_embedding)
        return x
