
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size,  embedding_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=-2, end_dim=-1)    # 将 patch 从 [B, C, H, W] 展平为 [B, C, H * W]
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
