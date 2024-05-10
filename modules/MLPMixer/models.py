import torch
from torch import nn

import einops
import flax.linen as nn2
import jax
import jax.numpy as jnp

import utils.logger.modellogger


def calculate_num_patches(input_shape, patch_size):
    """
    计算输入图像的 patch 数量
    Args:
        input_shape: 输入图像的形状，形式为 (B, C, H, W) 或 (C, H, W)，比如 (1, 3, 224, 224) 或 (3, 224, 224)
        patch_size: 输入图像的 patch 大小
    """
    h, w = input_shape[-2], input_shape[-1]     # 输入图像的大小
    assert h % patch_size == 0 & w % patch_size == 0, "Image size must be divisible by patch size."

    num_patches = (h // patch_size) * (w // patch_size)
    return num_patches


class MlpBlock(nn.Module):
    def __init__(self, d_x, d_mlp):
        """
        MLP-Mixer 的 MLP 模块
        :param d_x: 输入张量最后一层的维度
        :param d_mlp: MLP 的隐藏层维度
        """
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_x, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_x),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, d_x, num_patches, d_tokens_mlp=256, d_channels_mlp=2048):
        """
        MLP-Mixer 的 Mixer 模块
        Args:
            d_x (int): 输入张量最后一层的维度
            d_tokens_mlp (int): token 的 MLP 隐藏层维度
            d_channels_mlp (int): channel 的 MLP 隐藏层维度
        """
        super(MixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_x)
        self.token_mixing = MlpBlock(d_x=num_patches, d_mlp=d_tokens_mlp)   # 相当于广义的 depth-wise convolution

        self.norm2 = nn.LayerNorm(d_x)
        self.channel_mixing = MlpBlock(d_x=d_x, d_mlp=d_channels_mlp)       # 相当于广义的 1x1 convolution

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mixing(y).transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 input_shape,
                 patch_size,
                 d_hidden,
                 num_classes,
                 num_blocks=1,
                 d_tokens_mlp=256,
                 d_channels_mlp=2048):
        """
        MLP-Mixer 模型
        Args:
            input_shape (torch.Size | tuple): 输入图像的形状，形式为 (B, C, H, W) 或 (C, H, W)，比如 (1, 3, 224, 224) 或 (3, 224, 224)
            patch_size (int): 输入图像的 patch 大小，一般取 16 或 32
            d_hidden (int): 隐藏层的维度
            num_classes (int): 分类的类别数
            num_blocks (int): Mixer 模块的数量
            d_tokens_mlp (int): token 维度
            d_channels_mlp (int): channel 维度
        """
        super(MLPMixer, self).__init__()

        in_channels = input_shape[-3]                                 # 输入图像的通道数
        num_patches = calculate_num_patches(input_shape, patch_size)  # 输入图像的 patch 数量

        self.num_classes = num_classes

        # 使用卷积将图片分为多个 patch
        self.embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_hidden,
            kernel_size=patch_size,
            stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(d_x=d_hidden, num_patches=num_patches, d_tokens_mlp=d_tokens_mlp, d_channels_mlp=d_channels_mlp)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_hidden)
        if num_classes:
            self.fc = nn.Linear(d_hidden, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)

        # (B, C, H, W) -> (B, (H*W), C)
        x = self.flatten(x).transpose(1, 2)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.norm(x).mean(dim=1)
        if self.num_classes:
            x = self.fc(x)
        return x


if __name__ == '__main__':
    _input_shape = (1, 3, 224, 224)
    model = MLPMixer(input_shape=_input_shape,
                     patch_size=16,
                     d_hidden=512,
                     num_classes=1024,
                     num_blocks=8,
                     d_tokens_mlp=256,
                     d_channels_mlp=2048)

    x_input = torch.randn(_input_shape)
    x_output = model(x_input)
    print(x_output.shape)

    utils.logger.modellogger.log_model_params(model, x_input.shape)
