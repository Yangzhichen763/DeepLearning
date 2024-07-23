import torch
from torch import nn
import torch.nn.functional as F
from utils.log.model import *


def make_feature_layers(in_channels, out_channels, num_layers):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    return nn.Sequential(*layers)


class FCN(nn.Module):
    def __init__(self, d_hidden, num_classes, features=None):
        super(FCN, self).__init__()
        if features is not None:
            features = [64, 128, 256, 512]
        self.feature_layers = nn.Sequential(
            make_feature_layers(3, features[0], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_feature_layers(features[0], features[1], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_feature_layers(features[1], features[2], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_feature_layers(features[2], features[3], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(features[3], out_channels=d_hidden, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=d_hidden, out_channels=d_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=d_hidden, out_channels=num_classes, kernel_size=1),
        )

    def forward(self, x):
        x_shape = x.shape[-2:]
        x = self.feature_layers(x)
        x = self.classifier(x)
        x = F.interpolate(
            input=x,
            size=x_shape,
            mode='bilinear',
            align_corners=True)
        return x


if __name__ == '__main__':
    model = FCN(d_hidden=1024, num_classes=21, features=[64, 128, 256, 512])
    x_input = torch.randn(1, 3, 256, 256)
    y_pred = model(x_input)
    print(y_pred.shape)

    log_model_params(model, input_data=x_input)


