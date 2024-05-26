import torch
from torch import nn

import utils


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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size,  embedding_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=-2, end_dim=-1)    # 将 patch 展平为 [B, C, H * W]
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embedding_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches + 1, embedding_dim)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patcher(x).permute(0, 2, 1)                    # [B, C, H, W] ->[B, C, H * W] -> [B, H * W, C]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)   # [1, 1, embedding_dim] -> [B, 1, embedding_dim]
        print(cls_token.shape, x.shape)

        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_shape,
                 patch_size,
                 embed_dim,
                 dropout,
                 num_heads,
                 activation,
                 num_encoders,
                 num_classes):
        """
        Args:
            input_shape (torch.Size | tuple): 输入图像的形状，形式为 (B, C, H, W) 或 (C, H, W)，比如 (1, 3, 224, 224) 或 (3, 224, 224)
            patch_size (int): 输入图像的 patch 大小，一般取 16 或 32
            embed_dim (int): 输入 patch 的嵌入维度
            dropout: dropout 率，一般取 0.1
            num_heads (int): multi-head attention 的头数，一般取 6
            activation (str): transformer 的激活函数，能被 F. 出来的函数名，比如 "gelu"（即 F.gelu） 或 "relu"（即 F.relu）
            num_encoders (int): transformer 的编码器层数，一般取 6
            num_classes (int): 分类器的输出维度
        """
        super(VisionTransformer, self).__init__()

        in_channels = input_shape[-3]                                 # 输入图像的通道数
        num_patches = calculate_num_patches(input_shape, patch_size)  # 输入图像的 patch 数量

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
                                d_model=embed_dim,
                                nhead=num_heads,
                                dropout=dropout,
                                activation=activation,
                                batch_first=True,
                                norm_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)
        x = self.MLP(x[:, 0, :])
        return x


if __name__ == '__main__':
    _input_shape = (1, 3, 224, 224)
    _num_patches = calculate_num_patches(input_shape=_input_shape, patch_size=16)
    model = VisionTransformer(input_shape=_input_shape, patch_size=16, embed_dim=768, dropout=0.1,
                              num_heads=6, activation='gelu', num_encoders=6, num_classes=10)

    x_input = torch.randn(size=_input_shape)
    y_pred = model(x_input)
    print(y_pred.shape)

    # classify_utils.logger.log_model_params(model, x_input.shape)

