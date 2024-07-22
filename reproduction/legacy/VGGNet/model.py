import torch
import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    """
    论文地址：https://arxiv.org/abs/1409.1556
    团队主页：https://www.robots.ox.ac.uk/~vgg/research/very_deep/
    """

    def __init__(self, feature_layers, num_features=4096, num_classes=1000, init_weights=True):
        """
        :param feature_layers: VGG 网络特征层的结构
        :param num_classes: 分类个数
        """
        super(VGG, self).__init__()
        self.feature_layers = feature_layers            # VGG 网络特征层的结构
        # AdaptiveAvgPool2d 和 AvgPool2d 的区别是自适应平均池化层根据设置的输出信号尺寸以及输入信号的尺寸，自动计算池化层的大小以及移动步长
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))    # VGG 全连接层之前有 7x7 的平均池化层
        self.flattener = nn.Flatten()                   # 将特征层输出的张量压平到一维
        self.classifier = nn.Sequential(                # VGG 全连接层，用于分类
            nn.Linear(7 * 7 * 512, num_features),           # 使用 4096 个 7 * 7 * 512 的 filter 进行卷积，结果形状为 1 * 1 * 4096
            nn.ReLU(inplace=True),                          # 在全连接层后添加激活函数可以增加非线性表达，inplace=True 可以节省内存空间
            nn.Dropout(),                                   # Dropout 防止过拟合
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(num_features, num_classes),           # 全连接到分类结果
        )

        if init_weights:
            self.initialize()

    def forward(self, x):
        x = self.feature_layers(x)      # 特征提取，根据不同的 Configuration 选取不同的网络结构
        x = self.avg_pool(x)            # 全连接层前的池化
        x = self.flattener(x)           # 将张量维度压平到一维，以便全连接层计算
        x = self.classifier(x)          # 全连接层以及分类
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# VGG 的网络结构配置
configurations = {
    'A':        [64, 'max_pool', 128, 'max_pool', 256, 256, 'max_pool', 512, 512, 'max_pool', 512, 512],
    'A-LRN':    [64, 'max_pool', 128, 'LRN', 'max_pool', 256, 256, 'max_pool', 512, 512, 'max_pool', 512, 512],
    'B':        [64, 64, 'max_pool', 128, 128, 'max_pool', 256, 256, 'max_pool', 512, 512, 'max_pool', 512, 512],
    'C':        [64, 64, 'max_pool', 128, 128, 'max_pool', 256, 256, (1, 256), 'max_pool', 512, 512, (1, 512), 'max_pool', 512, 512, (1, 512)],
    'D':        [64, 64, 'max_pool', 128, 128, 'max_pool', 256, 256, 256, 'max_pool', 512, 512, 512, 'max_pool', 512, 512, 512],
    'E':        [64, 64, 'max_pool', 128, 128, 'max_pool', 256, 256, 256, 256, 'max_pool', 512, 512, 512, 512, 'max_pool', 512, 512, 512, 512],
}
configurationMap = {
    'max_pool': [nn.MaxPool2d(kernel_size=2, stride=2)],
    'LRN':      [nn.LocalResponseNorm(2)],
}


def make_feature_layer(configuration):
    """
    根据配置序列，生成 VGG 的网络特征层
    :param configuration: 配置序列，为 configurations 键值对中的 value
    :return: VGG 的网络特征层
    """
    feature_layers = []
    in_channels = 3
    for key in configuration:
        if configurationMap.__contains__(key):  # 添加非卷积层
            feature_layers += configurationMap[key]
        elif isinstance(key, tuple):            # 添加指定卷积核大小的卷积层
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=key[-1], kernel_size=key[0], padding=1)
            feature_layers += [conv2d, nn.ReLU()]
            in_channels = key[-1]
        else:                                   # 添加指定卷积核大小为 3 的卷积层
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=key, kernel_size=3, padding=1)
            feature_layers += [conv2d, nn.ReLU()]
            in_channels = key

    return nn.Sequential(*feature_layers)


def get_vgg(configuration_code, num_features=None, num_classes=1000):
    """
    生成指定 VGG 网络
    :param configuration_code: 配置序列码，可以输入：A、A-LRN、B、C、D、E
    :param num_features: 全连接层的输入维度
    :param num_classes: 分类数量
    :return: VGG 网络结构
    """
    if num_features is None:
        num_features = 16 * num_classes
    configuration = configurations[configuration_code]
    feature_layer = make_feature_layer(configuration)
    _model = VGG(feature_layer, num_features, num_classes)
    return _model


def VGG11(num_classes=1000, pretrained=False):
    if pretrained:
        return models.vgg11(pretrained=True)
    else:
        return get_vgg('A', num_classes)


def VGG13(num_classes=1000, pretrained=False):
    if pretrained:
        return models.vgg13(pretrained=True)
    else:
        return get_vgg('B', num_classes)


def VGG16(num_classes=1000, pretrained=False):
    if pretrained:
        return models.vgg16(pretrained=True)
    else:
        return get_vgg('D', num_classes)


def VGG19(num_classes=1000, pretrained=False):
    if pretrained:
        return models.vgg19(pretrained=True)
    else:
        return get_vgg('E', num_classes)


if __name__ == '__main__':
    # 测试 VGG16
    model = VGG16()
    print(model)

    x_input = torch.randn(1, 3, 224, 224)
    x_output = model(x_input)
    print(x_output.shape)


