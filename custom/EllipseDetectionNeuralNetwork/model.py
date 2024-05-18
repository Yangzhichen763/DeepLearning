import torch
from torch import nn

from torchvision.models import resnet18
from utils.logger import *


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


class EllipseDetectionNetwork(nn.Module):
    def __init__(
            self,
            input_shape,
            dim_features=None,
            num_layers_in_features=None,
            dim_classifiers=None,
            dim_output=5,
            num_output=64,
            dropout=0.1,
            device='cuda'):
        super(EllipseDetectionNetwork, self).__init__()
        if dim_features is None:
            dim_features = [64, 128, 256, 512, 4096]
        if num_layers_in_features is None:
            num_layers_in_features = [2, 2, 3, 3, 2]
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
            nn.Linear(in_features=in_features, out_features=dim_classifiers[-1], device=device),
        )
        self.classifiers = nn.Sequential(*classifiers)

        self.outputs = []
        # 如果 dim_output = 5，则输出为 x, y, w, h, angle, score
        for i in range(dim_output + 1):
            self.outputs.append(
                nn.Linear(
                    in_features=dim_classifiers[-1],
                    out_features=num_output,
                    device=device)
            )


    def forward(self, x):
        x = self.feature_layers(x)
        x = self.flatten(x)
        x = self.classifiers(x)

        y = []
        for output in self.outputs:
            y.append(output(x))
        x = torch.stack(y, dim=1).permute(0, 2, 1)
        return x


if __name__ == '__main__':
    _input_shape = (8, 1, 256, 256)
    model = EllipseDetectionNetwork(
        input_shape=_input_shape,
        dim_features=[8, 16, 32, 64, 128],
        num_layers_in_features=[2, 2, 3, 3, 2],
        dim_classifiers=[32, 16],
        device='cuda'
    )
    x_input = torch.randn(*_input_shape)

    log_model_params(model, _input_shape)


