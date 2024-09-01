from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neck import Neck


class FPNNeck(Neck):
    def __init__(
        self, *,
        backbone_out_channels: list[int],
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,

        out_levels: list[int] = None,
        interpolate_mode: str = "nearest",
        fuse_type: str = "avg"
    ):
        """
        Args:
            backbone_out_channels (list[int]): list of output channels of each feature map from the backbone
            out_channels (int): number of output channels for each convolutional layer
            kernel_size (int, optional): kernel size of the convolutional layer. Defaults to 1.
            stride (int, optional): stride of the convolutional layer. Defaults to 1.
            padding (int, optional): padding of the convolutional layer. Defaults to 0.
            out_levels (list[int], optional): list of output levels to be used for the FPN. Defaults to None.
            interpolate_mode (str, optional): interpolation mode for up-sampling. Defaults to "nearest".
            fuse_type (str, optional): type of fusing method for combining the features from different levels. Defaults to "avg".
        """
        super(FPNNeck, self).__init__()
        # 对每个特征图进行卷积，使得通道数和 backbone 输出特征图的通道数一致
        self.convs = nn.ModuleList()
        for dim in backbone_out_channels:
            current = nn.Sequential()
            current.add_module(
                name="conv",
                module=nn.Conv2d(
                    in_channels=dim,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ),
            )

            self.convs.append(current)

        # 进行特征融合的层
        if out_levels is None:
            out_levels = list(range(len(backbone_out_channels)))
        self.out_levels = out_levels
        self.interpolate_mode = interpolate_mode

        # 融合特征图使用平均还是求和
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        FPN 特征融合过程中的前向传播
        Args:
            xs (list[torch.Tensor]): list of input tensors from the backbone

        Returns:
            list[torch.Tensor]: list of output tensors for each output level
        """
        assert len(xs) == len(self.convs), "Number of input tensors and convolutional layers should be the same"

        out: list[Optional[torch.Tensor]] = [None] * len(xs)
        prev_features: Optional[torch.Tensor] = None
        n = len(xs) - 1
        for i in range(n, -1, -1):    # [n, n-1, ..., 0]
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if prev_features is None:
                prev_features = lateral_features
            else:
                # 提高上一个特征图的分辨率（提高 2 倍）
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.interpolate_mode,
                    align_corners=None if self.interpolate_mode == "nearest" else False,
                    antialias=False
                )
                # 融合特征图
                if i in self.out_levels:
                    prev_features = lateral_features + top_down_features
                    if self.fuse_type == "avg":
                        prev_features /= 2
                else:
                    prev_features = top_down_features

            x_out = prev_features
            out[i] = x_out
        return out


if __name__ == "__main__":
    print([i for i in range(10, -1, -1)])
