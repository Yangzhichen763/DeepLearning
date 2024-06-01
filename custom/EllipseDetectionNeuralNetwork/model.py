import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import resnet18
from utils.logger import *
from modules.residual import ResNetEncoder, ResNeXtEncoder, BasicBlock, Bottleneck
from modules.fpn import FeaturePyramidNetwork


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ConvBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class EllipseDetectionNetwork_AlexNet(nn.Module):
    def __init__(
            self,
            input_shape,
            num_layers_in_features=None,
            dim_features=None,
            dim_classifiers=None,
            dim_output=5,
            num_output=64,
            dropout=0.1,
            device='cuda'):
        super(EllipseDetectionNetwork_AlexNet, self).__init__()
        if num_layers_in_features is None:
            num_layers_in_features = [2, 2, 3, 3, 2]
        if dim_features is None:
            dim_features = [64, 128, 256, 512, 4096]
        if dim_classifiers is None:
            dim_classifiers = [64, 16]

        in_channels, h, w = input_shape[-3], input_shape[-2], input_shape[-1]
        feature_layers = []
        for i, (feature, num_layers) in enumerate(zip(dim_features, num_layers_in_features)):
            if i != 0:
                feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                h //= 2
                w //= 2
            feature_layers.append(ConvBlock(in_channels, feature, num_layers))
            in_channels = feature
        self.feature_layers = nn.Sequential(*feature_layers)

        self.flatten = nn.Flatten()

        self.out_features = dim_classifiers[-1]
        classifiers = []
        in_features = in_channels * h * w
        for dim_classifier in dim_classifiers[:-1]:
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=in_features, out_features=dim_classifier, device=device),
                    nn.ReLU()
                )
            )
            in_features = dim_classifier
        classifiers.append(
            nn.Linear(in_features=in_features, out_features=self.out_features, device=device),
        )
        self.classifiers = nn.Sequential(*classifiers)

        # 如果 dim_output = 5，则输出为 x, y, w, h, angle
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    in_features=self.out_features,
                    out_features=num_output,
                    device=device)
            )
            for _ in range(dim_output)
        ])


    def forward(self, x):
        x = self.feature_layers(x)
        x = self.flatten(x)
        x = self.classifiers(x)

        y = []
        for output in self.outputs:
            y.append(output(x))
        x = torch.stack(y, dim=1).permute(0, 2, 1)
        return x


class EllipseDetectionNetwork_ResNet_v1(nn.Module):
    def __init__(
            self,
            center_encoder=None,
            size_encoder=None,
            dim_classifiers=None,
            dim_output=5,
            num_output=64,
            dropout=0.1,
            device='cuda'):
        """

        Args:
            center_encoder (ResNetEncoder | ResNeXtEncoder):
            size_encoder (ResNetEncoder | ResNeXtEncoder):
            dim_classifiers:
            dim_output:
            num_output:
            dropout:
            device:
        """
        super(EllipseDetectionNetwork_ResNet_v1, self).__init__()
        if dim_classifiers is None:
            dim_classifiers = [32, 16]

        self.center_encoder = center_encoder
        self.size_encoder = size_encoder

        self.out_features = dim_classifiers[-1]
        classifiers = []
        in_features = center_encoder.out_channels + size_encoder.out_channels
        for dim_classifier in dim_classifiers[:-1]:
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=in_features, out_features=dim_classifier, device=device),
                    nn.ReLU()
                )
            )
            in_features = dim_classifier
        classifiers.append(
            nn.Linear(in_features=in_features, out_features=2 * self.out_features, device=device),
        )
        self.classifiers = nn.Sequential(*classifiers)

        # 如果 dim_output = 5，则输出为 x, y, w, h, angle
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    in_features=self.out_features,
                    out_features=num_output,
                    device=device)
            )
            for _ in range(dim_output)
        ])

    def forward(self, x):
        # 假设 dim_features = [16, 32, 64, 128, 256]
        # [B, C, H, W] -> [B, 256, H / 64, W / 64]，其中 256 = dim_features[-1]，64 = 2 ** 5
        x_center = self.center_encoder(x)
        x_size = self.size_encoder(x)

        # [B, 256, H / 64, W / 64] -> [B, 256, 1, 1] -> [B, 256]
        x_center = torch.flatten(F.adaptive_max_pool2d(x_center, (1, 1)), start_dim=1)
        x_size = torch.flatten(F.adaptive_max_pool2d(x_size, (1, 1)), start_dim=1)

        # [B, 256] -> [B, 256 + 256]
        x = torch.cat([x_center, x_size], dim=-1)
        # 假设 dim_classifiers = [64, 16]
        # [B, 256 + 256] -> [B, 64] -> [B, 2 * 16]
        x = self.classifiers(x)
        x_center, x_size = torch.split(x, [self.out_features, self.out_features], dim=-1)

        # [B, 16] -> [B, 64]
        cx = self.outputs[0](x_center)
        cy = self.outputs[1](x_center)
        w = self.outputs[2](x_size)
        h = self.outputs[3](x_size)
        # for param in self.outputs[4][1].parameters():
        #     param.data = param.data.sin().asin()
        angle = self.outputs[4](x_size)
        # 5 * [B, 64] -> [B, 5, 64] -> [B, 64, 5]
        x = torch.stack([cx, cy, w, h, angle], dim=1).permute(0, 2, 1)

        return x


class EllipseDetectionNetwork_ResNet_v2(nn.Module):
    def __init__(
            self,
            encoder=None,
            dim_classifiers=None,
            dim_output=5,
            num_output=64,
            dropout=0.1,
            device='cuda'):
        """

        Args:
            center_encoder (ResNetEncoder):
            size_encoder (ResNetEncoder):
            angle_encoder (ResNetEncoder):
            dim_classifiers:
            dim_output:
            num_output:
            dropout:
            device:
        """
        super(EllipseDetectionNetwork_ResNet_v2, self).__init__()
        if dim_classifiers is None:
            dim_classifiers = [32, 16]

        self.encoder = encoder

        self.out_features = dim_classifiers[-1]
        classifiers = []
        in_features = encoder.out_channels
        for dim_classifier in dim_classifiers[:-1]:
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=in_features, out_features=dim_classifier, device=device),
                    nn.ReLU()
                )
            )
            in_features = dim_classifier
        classifiers.append(
            nn.Linear(in_features=in_features, out_features=self.out_features, device=device),
        )
        self.classifiers = nn.Sequential(*classifiers)

        # 如果 dim_output = 5，则输出为 x, y, w, h, angle
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    in_features=self.out_features,
                    out_features=num_output,
                    device=device)
            )
            for _ in range(dim_output)
        ])

    def forward(self, x):
        # 假设 dim_features = [16, 32, 64, 128, 256]
        # [B, C, H, W] -> [B, 256, H / 64, W / 64]，其中 256 = dim_features[-1]，64 = 2 ** 5
        x = self.encoder(x)

        # [B, 256, H / 64, W / 64] -> [B, 256, 1, 1] -> [B, 256]
        x = torch.flatten(F.adaptive_max_pool2d(x, (1, 1)), start_dim=1)
        # 假设 dim_classifiers = [32, 16]
        # [B, 256] -> [B, 32] -> [B, 16]
        x = self.classifiers(x)

        # 5 * [B, 16] -> 5 * [B, 64]
        y = [output(x) for output in self.outputs]
        # 5 * [B, 64] -> [B, 5, 64] -> [B, 64, 5]
        x = torch.stack(y, dim=1).permute(0, 2, 1)
        return x


class EllipseDetectionNetwork_FPN(nn.Module):
    def __init__(self, encoder, dim_classifiers, dim_output, num_output, dropout, device):
        super(EllipseDetectionNetwork_FPN, self).__init__()
        self.encoder = encoder
        self.dim_classifiers = dim_classifiers
        self.dim_output = dim_output
        self.fpn = FeaturePyramidNetwork()


if __name__ == '__main__':
    _input_shape = (2, 3, 64 * 5, 64 * 5)
    _in_channels = _input_shape[1]
    x_input = torch.randn(*_input_shape)

    # ============= ResNet v1.0
    _dim_classifiers = [512, 256]
    _center_encoder = ResNetEncoder(
        _in_channels,
        BasicBlock,
        num_blocks=[2, 2, 3, 3, 2],
        dim_hidden=[16, 32, 64, 128, 256])
    _size_encoder = ResNetEncoder(
        _in_channels,
        Bottleneck,
        num_blocks=[3, 4, 6, 3],
        dim_hidden=[16, 32, 64, 128])

    model = EllipseDetectionNetwork_ResNet_v1(
        center_encoder=_center_encoder,
        size_encoder=_size_encoder,
        dim_classifiers=_dim_classifiers,
        device='cuda'
    )

    log_model_params(model, _input_shape)
    exit()

    # ============= ResNet v2.0
    _dim_classifiers = [32, 16]
    _encoder = ResNetEncoder(
        _in_channels,
        num_blocks=[3, 4, 6, 3],
        dim_hidden=[32, 64, 128, 256])
    model = EllipseDetectionNetwork_ResNet_v2(
        encoder=_encoder,
        dim_classifiers=_dim_classifiers,
        device='cuda'
    )

    log_model_params(model, _input_shape)

    # ============= AlexNet
    model = EllipseDetectionNetwork_AlexNet(
        input_shape=_input_shape,
        dim_features=[16, 32, 64, 128, 256],
        num_layers_in_features=[2, 2, 3, 3, 2],
        dim_classifiers=[32, 16],
        device='cuda'
    )

    log_model_params(model, _input_shape)


