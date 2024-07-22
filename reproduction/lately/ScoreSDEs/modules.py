
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.activation import Swish


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier projection module.
    """
    def __init__(self, embedding_dim, scale=30):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2) * scale)
        self.weight.requires_grad = False

    def forward(self, t):
        t_projection = t[:, None] * self.weight[None, :] * 2 * math.pi
        return torch.cat([torch.sin(t_projection), torch.cos(t_projection)], dim=-1)


class Dense(nn.Module):
    """
    先进行全连接，然后改变形状 [N] -> [N, 1, 1]
    全过程：[in_channels] -> [out_channels] -> [out_channels, 1, 1]
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, num_groups, **conv_kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, **conv_kwargs)
        self.dense = Dense(embedding_dim, out_channels)     # -> nn.Linear()[..., None, None]
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x, time_embedding):
        x = self.conv(x)
        x += self.dense(time_embedding)
        x = self.norm(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, num_groups, **conv_kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, **conv_kwargs)
        self.dense = Dense(embedding_dim, out_channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x, time_embedding):
        x = self.conv(x)
        x += self.dense(time_embedding)
        x = self.norm(x)
        return x


class ScoreUNet(nn.Module):
    sigma = 25

    # noinspection PyPep8Naming
    def __init__(self, marginal_prob_std, embedding_dim=256, feature_dims=None):
        super(ScoreUNet, self).__init__()
        if feature_dims is None:
            feature_dims = [64, 128, 256, 512]

        self.embedding = GaussianFourierProjection(256)

        self.down_sample_layers = nn.ModuleList([
            DownSample(1, feature_dims[0], embedding_dim, 4, stride=1, bias=False),
            DownSample(feature_dims[0], feature_dims[1], embedding_dim, 32, stride=2, bias=False),
            DownSample(feature_dims[1], feature_dims[2], embedding_dim, 32, stride=2, bias=False),
            DownSample(feature_dims[2], feature_dims[3], embedding_dim, 32, stride=2, bias=False)
        ])
        self.up_sample_layers = nn.ModuleList([
            UpSample(feature_dims[3], feature_dims[2], embedding_dim, 32, stride=2, bias=False),
            UpSample(feature_dims[2] * 2, feature_dims[1], embedding_dim, 32, stride=2, padding=1, bias=False),
            UpSample(feature_dims[1] * 2, feature_dims[0], embedding_dim, 32, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(feature_dims[0] * 2, 1, kernel_size=3, stride=1, bias=True)
        ])

        self.activation = Swish()
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        time_embedding = self.activation(self.embedding(t))

        # 下采样
        x1 = self.activation(self.down_sample_layers[0](x, time_embedding))
        x2 = self.activation(self.down_sample_layers[1](x1, time_embedding))
        x3 = self.activation(self.down_sample_layers[2](x2, time_embedding))
        x4 = self.activation(self.down_sample_layers[3](x3, time_embedding))

        # 上采样
        x = self.activation(self.up_sample_layers[0](x4, time_embedding))
        x = self.activation(self.up_sample_layers[1](torch.cat([x, x3], dim=1), time_embedding))
        x = self.activation(self.up_sample_layers[2](torch.cat([x, x2], dim=1), time_embedding))
        x = self.up_sample_layers[3](torch.cat([x, x1], dim=1), time_embedding)

        x = x.sigmoid() / self.marginal_prob_std(t, self.sigma, x.device)[..., None, None, None]

        return x

