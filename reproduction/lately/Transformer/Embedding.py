import torch
from torch import nn
from modules.embedding import TokenEmbedding, SinusoidalPositionalEmbedding1d


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout, device=None):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalPositionalEmbedding1d(d_model, max_len, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(x)
        x = self.dropout(token_embedding + positional_embedding)
        return x
