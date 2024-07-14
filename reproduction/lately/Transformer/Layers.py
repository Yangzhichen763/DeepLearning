import torch
from torch import nn
from lately.Transformer.Modules import MultiheadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn1 = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_residual = x
        x, _ = self.attention(x, x, x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + x_residual)

        x_residual = x
        x = self.ffn1(x)
        x = self.dropout2(x)
        x = self.norm2(x + x_residual)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.attention2 = MultiheadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn1 = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, tgt_mask=None, src_mask=None):
        x_residual = dec_input
        x, _ = self.attention1(dec_input, dec_input, dec_input, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x + x_residual)

        if enc_output is not None:
            x_residual = x
            x, _ = self.attention2(x, enc_output, enc_output, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + x_residual)

        x_residual = x
        x = self.ffn1(x)
        x = self.dropout3(x)
        x = self.norm3(x + x_residual)

        return x