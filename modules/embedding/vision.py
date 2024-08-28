
import torch
import torch.nn as nn


class GlobalPatchEmbedding(nn.Module):
    """
    patch pattern:
        X O X O
        O O O O
        X O X O
        O O O O
    """
    def __init__(self, in_channels: int, image_size: tuple[int, int], patch_size: int, embedding_dim: int):
        super(GlobalPatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dilation = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 dilation=self.dilation)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            embedding: [B, P, P, C*H*W//P^2], p=patch_size
        """
        x = self.patcher(x).permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, P, P, C*H*W//P^2], p=patch_size
        return x


class LocalPatchEmbedding(nn.Module):
    """
    patch pattern:
        X X O O
        X X O O
        O O O O
        O O O O
    """
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int):
        super(LocalPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            embedding: [B, P, P, C*H*W//P^2], P=patch_size
        """
        x = self.patcher(x).permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, P, P, C*H*W//P^2], P=patch_size
        return x


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int,
            embedding_dim: int,
            kernel_size: tuple[int, int],
            stride: tuple[int, int],
            padding: tuple[int, int]
    ):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            embedding: [B, P, P, C*H*W//P^2], P=patch_size
        """
        x = self.patcher(x).permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, P, P, C*H*W//P^2], P=patch_size
        return x


if __name__ == '__main__':

    x_input = torch.randn(size=(2, 3, 224, 224))

    # GlobalPatchEmbedding
    model = GlobalPatchEmbedding(in_channels=3, image_size=(224, 224), patch_size=16, embedding_dim=768)
    out = model(x_input)
    print("GlobalPatchEmbedding out.shape:", out.shape)

    # LocalPatchEmbedding
    model = LocalPatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)
    out = model(x_input)
    print("LocalPatchEmbedding out.shape:", out.shape)
