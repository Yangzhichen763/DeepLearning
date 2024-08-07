from .parts import *

from utils.log import *


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(UNet, self).__init__()
        self.in_conv = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out_conv = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x


class UNetBilinear(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(UNetBilinear, self).__init__()
        self.down1 = DownBilinear(in_channels, 64)
        self.down2 = DownBilinear(64, 128)
        self.down3 = DownBilinear(128, 256)
        self.down4 = DownBilinear(256, 512)
        self.mid_conv = DoubleConv(512, 512)
        self.up1 = UpBilinear(512, 256)
        self.up2 = UpBilinear(256, 128)
        self.up3 = UpBilinear(128, 64)
        self.up4 = UpBilinear(64, 64)
        self.out_conv = OutConv(64, num_classes)

    def forward(self, x):
        x, x1 = self.down1(x)
        print(x.shape, x1.shape)
        x, x2 = self.down2(x)
        print(x.shape, x2.shape)
        x, x3 = self.down3(x)
        print(x.shape, x3.shape)
        x, x4 = self.down4(x)
        print(x.shape, x4.shape)
        x = self.mid_conv(x)
        print(x.shape)
        x = self.up1(x, x4)
        print(x.shape)
        x = self.up2(x, x3)
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        print(x.shape)
        x = self.out_conv(x)
        print(x.shape)
        return x


class UNetCustom(nn.Module):
    def __init__(self, in_channels, num_classes, feature_dims=None):
        """
        Args:
            in_channels (int):
            num_classes (int):
            feature_dims (list): 输入的参数有 n 个，则会进行 n - 1 次下采样和上采样
        """
        super(UNetCustom, self).__init__()
        if feature_dims is None:
            feature_dims = [64, 128, 256, 512, 1024]
        self.in_conv = InConv(in_channels, feature_dims[0])
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.out_conv = OutConv(feature_dims[0], num_classes)

        in_channels = feature_dims[0]
        for feature in feature_dims[1:]:
            self.down_convs.append(Down(in_channels, feature))
            in_channels = feature

        for feature in reversed(feature_dims[:-1]):
            self.up_convs.append(Up(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        skip_connections = []
        x = self.in_conv(x)

        for down_conv in self.down_convs:
            skip_connections.append(x)
            x = down_conv(x)

        for x_residual, up_conv in zip(reversed(skip_connections), self.up_convs):
            x = up_conv(x, x_residual)

        x = self.out_conv(x)
        return x


class UNetBilinearCustom(nn.Module):
    def __init__(self, in_channels, num_classes, features=None):
        """
        Args:
            in_channels (int):
            num_classes (int):
            features (list): 输入的参数有 n 个，则会进行 n 次下采样和上采样
        """
        super(UNetBilinearCustom, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.down_convs = nn.ModuleList()
        self.mid_conv = DoubleConv(features[-1], features[-1])
        self.up_convs = nn.ModuleList()
        self.out_conv = OutConv(features[0], num_classes)

        for feature in features:
            self.down_convs.append(DownBilinear(in_channels, feature))
            in_channels = feature

        for feature in reversed([features[0]] + features[:-1]):
            self.up_convs.append(UpBilinear(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        skip_connections = []

        for down_conv in self.down_convs:
            x, residual = down_conv(x)
            skip_connections.append(residual)

        x = self.mid_conv(x)

        for x_residual, up_conv in zip(reversed(skip_connections), self.up_convs):
            x = up_conv(x, x_residual)

        x = self.out_conv(x)
        return x


def UNet_custom_light(num_classes=21, bilinear=True):
    if bilinear:
        return UNetBilinearCustom(3, num_classes, features=[4, 8, 16, 32, 64])
    else:
        return UNetCustom(3, num_classes, feature_dims=[4, 8, 16, 32, 64])


if __name__ == '__main__':
    _model = UNet_custom_light()
    x_input = torch.randn(1, 3, 256, 256)
    x_output = _model(x_input)
    print(x_output.shape)

    log_model_params(_model, input_size=x_input.shape)
