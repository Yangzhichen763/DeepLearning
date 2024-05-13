import torch
from torch import nn
from lately.Transformer.Embedding import TransformerEmbedding
from lately.Transformer.Layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, env_voc_size, max_len, d_model, d_ffn, n_heads, n_layer, dropout, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(d_model, max_len, env_voc_size, dropout, device)

        self.layer = nn.ModuleList(
            [EncoderLayer(d_model, d_ffn, n_heads, dropout) for _ in range(n_layer)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layer:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, env_voc_size, max_len, d_model, d_ffn, n_heads, n_layer, dropout, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(d_model, max_len, env_voc_size, dropout, device)

        self.layer = nn.ModuleList(
            [DecoderLayer(d_model, d_ffn, n_heads, dropout) for _ in range(n_layer)])

        self.fc = nn.Linear(d_model, env_voc_size)

    def forward(self, dec_input, enc_output, tgt_mask=None, src_mask=None):
        dec_input = self.embedding(dec_input)
        for layer in self.layer:
            dec_input = layer(dec_input, enc_output, tgt_mask, src_mask)

        dec_input = self.fc(dec_input)

        return dec_input


def make_pad_mask(q, k, pad_idx_q, pad_idx_k):
    len_q, len_k = q.shape[1], k.shape[1]

    # (Batch, Time, len_q, len_k)
    q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
    q = q.repeat(1, 1, 1, len_k)

    k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
    k = k.repeat(1, 1, len_q, 1)

    mask = q & k
    return mask


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, dst_pad_idx, enc_voc_size, dec_voc_size, max_len, d_model, d_ffn, n_heads, n_layer, dropout, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, d_ffn, n_heads, n_layer, dropout, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, d_ffn, n_heads, n_layer, dropout, device)

        self.src_pad_idx = src_pad_idx
        self.dst_pad_idx = dst_pad_idx
        self.device = device

    def make_casual_mask(self, q, k):
        len_q, len_k = q.shape[1], k.shape[1]
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self, src, tgt):
        src_mask = make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        tgt_mask = make_pad_mask(tgt, tgt, self.dst_pad_idx, self.dst_pad_idx) * self.make_casual_mask(tgt, tgt)
        src_tgt_mask = make_pad_mask(tgt, src, self.tgt_pad_idx, self.src_pad_idx)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_tgt_mask)
        return dec_output
