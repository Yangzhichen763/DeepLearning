
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from modules.attention import VisionAttention
from modules.activation import Swish


class TimeEmbeddingProjection(nn.Module):
    """
    Time embedding projection module.
    """
    # noinspection PyPep8Naming
    def __init__(self, T, embedding_dim, dim):
        assert embedding_dim % 2 == 0
        super().__init__()
        position = torch.arange(T).float()                  # [T]
        embedding = torch.exp(
            torch.arange(0, embedding_dim, step=2)
            / embedding_dim * math.log(10000)
        )                                                   # [d_model // 2]
        embedding = position[:, None] * embedding[None, :]  # [T, 1] * [1, d_model // 2] -> [T, d_model // 2]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(embedding),
            nn.Linear(embedding_dim, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        embedding = self.time_embedding(t)
        return embedding


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, time_embedding):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, d_model)
        self.attention = VisionAttention(d_model)

    def forward(self, x):
        x_norm = self.group_norm(x)
        output, attention = self.attention(x_norm)

        return x + output


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.time_embedding_projection = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_channels),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.time_embedding_projection(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, t, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbeddingProjection(t, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_channels=now_ch, out_channels=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_channels=chs.pop() + now_ch, out_channels=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        t=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)

