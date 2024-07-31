import torch
from torch import nn
from torch.nn import functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    [B, C, H, W]
     ->
     [B, dim_smooth, H/4, W/4]
     [B, dim_smooth, H/8, W/8]
     [B, dim_smooth, H/16, W/16]
     [B, dim_smooth, H/32, W/32]
    """
    def __init__(
            self,
            in_channels=3,
            bottleneck=None,
            num_blocks=None,
            dim_planes=None,
            dim_smooth=256):
        """

        Args:
            in_channels:
            bottleneck: 可以取 resnet bottleneck 或者 inception bottleneck
            num_blocks:
            dim_planes:
            dim_smooth:
        """
        super(FeaturePyramidNetwork, self).__init__()
        if num_blocks is None:
            num_blocks = [3, 4, 6, 3]
        if dim_planes is None:
            dim_planes = [64, 128, 256, 512]
        strides = [1 if i == 0 else 2 for i in range(len(dim_planes))]
        self.in_planes = dim_planes[0]

        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Bottom-up layers
        self.layers = nn.ModuleList([
            self._make_layer(bottleneck, num_plane, num_block, stride)
            for (num_plane, num_block, stride) in zip(dim_planes, num_blocks, strides)
        ])

        # Top layer
        self.top_layer = nn.Conv2d(self.in_planes, dim_smooth, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(dim_smooth, dim_smooth, kernel_size=3, stride=1, padding=1)
            for _ in range(len(dim_planes))
        ])

        # Lateral layers
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(num_plane * bottleneck.expansion, dim_smooth, kernel_size=1, stride=1, padding=0)
            for num_plane in reversed(dim_planes[:-1])
        ])

    def _make_layer(self, bottleneck, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != bottleneck.expansion * planes:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(bottleneck.expansion * planes)
            )

        layers = [bottleneck(self.in_planes, planes, stride, down_sample)]
        self.in_planes = planes * bottleneck.expansion
        for i in range(1, blocks):
            layers.append(bottleneck(self.in_planes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _upsample(x, h, w):  # upsample use 'bilinear' interpolate
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    @staticmethod
    def _upsample_add(x, y):
        _, _, h, w = y.shape
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layers[0](c1)
        c3 = self.layers[1](c2)
        c4 = self.layers[2](c3)
        c5 = self.layers[3](c4)

        # Top-down
        p5 = self.top_layer(c5)
        p4 = self._upsample_add(p5, self.lateral_layers[0](c4))
        p3 = self._upsample_add(p4, self.lateral_layers[1](c3))
        p2 = self._upsample_add(p3, self.lateral_layers[2](c2))

        # Smooth
        p4 = self.smooth_layers[0](p4)
        p3 = self.smooth_layers[1](p3)
        p2 = self.smooth_layers[2](p2)

        return p2, p3, p4, p5


if __name__ == '__main__':
    from utils.log.model import log_model_params

    model = FeaturePyramidNetwork(
        in_channels=3,
        num_blocks=[3, 4, 6, 3],
        dim_planes=[16, 32, 64, 128],
        dim_smooth=64)
    x_input = torch.randn(2, 3, 224, 224)

    log_model_params(model, input_data=x_input)
