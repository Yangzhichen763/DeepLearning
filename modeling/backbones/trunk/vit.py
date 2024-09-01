import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from modules.attention.Attention import MultiheadSelfAttention
from modeling.backbones.block import SelfAttentionBlock, ScalableModule
from modules.mlp import MLP
from modules.embedding.vision import LocalPatchEmbedding


class AttentionBlock(SelfAttentionBlock):
    def __init__(
        self,  *,
        dim: int,
        mlp_ratio: float = 4.0,
        num_heads: int = 8,
        attn_drop: float = 0.0,     # Attention 的注意力矩阵的 dropout
        proj_drop: float = 0.0,     # Attention 以及 MLP 的输出的 dropout
        drop_path: float = 0.0,     # Attention 以及 MLP 输出之后进行 DropPath 的 dropout 值
        norm_layer: [nn.Module, (nn.Module, nn.Module)] = nn.LayerNorm,
        **kwargs
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            attn=ScalableModule(
                block=MultiheadSelfAttention(
                    num_heads=num_heads,
                    d_model=dim,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    is_weighted=True,
                    any_bias=True
                ),
                dim=dim
            ),
            ffn=ScalableModule(
                block=MLP(
                    input_dim=dim,
                    hidden_dim=int(dim * mlp_ratio),
                    output_dim=dim,
                    num_layers=3,
                    drop_out=proj_drop,
                ),
                dim=dim
            ),
            drop_path_rate=drop_path,
            norm_layer=norm_layer,
            **kwargs
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x = super().forward(x, **kwargs)
        return x


class ViTPatchEmbedding(nn.Module):
    """
    Vision Transformer patch embedding
    """
    def __init__(self, image_size, patch_size, in_channels, embedding_dim, dropout=0.0):
        """
        Args:
            image_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channels (int): number of input channels
            embedding_dim (int): embedding dimension
            dropout (float): dropout rate
        """
        super(ViTPatchEmbedding, self).__init__()

        # 计算 num_patches
        def get_patch_grid(_image_size, _patch_size):
            is_image_size_tuple = isinstance(_image_size, tuple)
            is_patch_size_tuple = isinstance(_patch_size, tuple)
            if not is_image_size_tuple and not is_patch_size_tuple:
                return _image_size // _patch_size, _image_size // _patch_size
            if is_image_size_tuple and not is_patch_size_tuple:
                _patch_size = (_patch_size,) * len(_image_size)
            elif not is_image_size_tuple and is_patch_size_tuple:
                _image_size = (_image_size,) * len(_patch_size)

            if len(_image_size) != len(_patch_size):
                raise ValueError("image_size and patch_size must have the same number of dimensions")

            return tuple([_image_size[i] // _patch_size[i] for i in range(len(_image_size))])
        self.patch_grid = get_patch_grid(image_size, patch_size)
        self.num_patches = math.prod(self.patch_grid)
        self.embedding_dim = embedding_dim

        # 将图像分为 num_patches 个 patch_size 大小的块，然后将 patch 从 [B, H, W, C] 展平为 [B, H * W, C]
        self.patcher = LocalPatchEmbedding(in_channels, patch_size, embedding_dim)
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-2)  # 将 patch 从 [B, H, W, C] 展平为 [B, H * W, C]

        # cls_token（class token）的主要特点是：不基于图像内容，位置编码固定
        #   - class token 的输出用来预测类别，这种结构迫使 patch token 和 class token 之间传播信息
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embedding_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, self.num_patches + 1, embedding_dim)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

        trunc_normal_(self.position_embedding, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    # noinspection PyPep8Naming
    def interpolate_pos_encoding(self, x, H, W):
        """
        将 positional encoding 进行插值，使得其与 x 的大小相同
        Args:
            x (tensor): input tensor with shape [B, N+1, dim]
            H (int): height of the image
            W (int): width of the image

        Returns:
                output tensor with shape [B, N+1, dim]，其中 N+1 = num_patches + 1, dim = embedding_dim
        """
        num_patches = x.shape[1] - 1
        N = self.num_patches
        if num_patches == N and H == W:
            return self.position_embedding                   # [1, N+1, dim]

        dim = self.embedding_dim
        class_pos_embed = self.position_embedding[:, 0]      # class token      [1, 1, dim]
        patch_pos_embed = self.position_embedding[:, 1:]     # patch embedding  [1, N, dim]
        # 添加一个小的数值，以避免插值时出现浮点误差（由于插值是输入的 scale_factor 为 int 类型，H0 / sqrt(N) * sqrt(N) 可能不为 H0）
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        H0, W0 = self.patch_grid[0] + 0.1, self.patch_grid[1] + 0.1
        patch_pos_embed = nn.functional.interpolate(
            # [1, N, dim] -> [1, sqrt(N), sqrt(N), dim] -> [1, dim, sqrt(N), sqrt(N)]
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(H0 / math.sqrt(N), W0 / math.sqrt(N)),
            mode='bicubic',
        )                                                    # [1, dim, H0, W0]
        assert int(H0) == patch_pos_embed.shape[-2] and int(W0) == patch_pos_embed.shape[-1]

        # [1, dim, H0, W0] -> [1, H0, W0, dim] -> [1, H0 * W0, dim] == [1, N, dim]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # [1, 1, dim] + [1, N, dim] ==concat=> [1, N+1, dim]
        y = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return y

    # noinspection PyPep8Naming
    def forward(self, x):
        """
        Args:
            x (tensor): input tensor with shape [B, C, H, W]

        Returns:
            output tensor with shape [B, 1 + P * P, N]
        """
        B, C, H, W = x.shape
        x = self.flatten(self.patcher(x))             # [B, P, P, N] -> [B, P * P, N]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)   # [1, 1, embedding_dim] -> [B, 1, embedding_dim]

        x = torch.cat([cls_token, x], dim=1)   # [B, 1, N] + [B, P * P, N] ==concat=> [B, 1 + P * P, N]
        # 其中 interpolate_pos_encoding 是用于将 position_embedding 插值为 x 形状
        x = x + self.interpolate_pos_encoding(self.position_embedding, H, W)
        x = self.dropout(x)
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, *,
                 image_size=224, patch_size=16, in_channels=3, embedding_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embedding_dim

        self.patch_embedding = ViTPatchEmbedding(
            image_size=image_size, patch_size=patch_size,
            in_channels=in_channels, embedding_dim=embedding_dim,
            dropout=proj_drop)

        # 注意力块，随深度增加，drop_out 值逐渐增大（0 ~ drop_path_rate）
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度损失
        self.blocks = nn.ModuleList([                                            # 注意力块
            AttentionBlock(
                dim=embedding_dim, mlp_ratio=mlp_ratio, num_heads=num_heads,
                attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path_rate[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embedding_dim)

        # 权重初始化
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x):
        # position embedding
        x = self.patch_embedding(x)

        # transformer encoder
        for block in self.blocks:
            x = block(x)
        return x[:, 0]


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformerEncoder(
        patch_size=patch_size, embedding_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformerEncoder(
        patch_size=patch_size, embedding_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformerEncoder(
        patch_size=patch_size, embedding_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    x_input = torch.randn(2, 3, 224, 224)
    _model = vit_tiny()
    _output = _model(x_input)
    print(_output.shape)