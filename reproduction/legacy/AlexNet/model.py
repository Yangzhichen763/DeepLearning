import torch
import torch.nn as nn
from utils.log import *


class AlexNet(nn.Module):
    """
    论文地址：https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """
    
    def __init__(self, num_features=4096, num_classes=1000):
        """
        图像的输入形状为 [3, 227, 277]
        :param num_classes: 分类个数
        """
        super(AlexNet, self).__init__()
        # 图像的输入形状为 [3, 227, 227]
        self.net = nn.Sequential(  # [C, H, W]
            # C1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # [96, 55, 55]
            nn.ReLU(inplace=True),
            # LRN 对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # 论文中使用 size=5, alpha=0.0001, beta=0.75, k=2
            # 使用最大池化避免平均池化层的模糊效果，步长比核尺寸小提升了特征的丰富性、避免过拟合
            nn.MaxPool2d(kernel_size=3, stride=2),  # [96, 27, 27]

            # C2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),  # [256, 27, 27]
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [256, 13, 13]

            # C3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),  # [384, 13, 13]
            nn.ReLU(inplace=True),

            # C4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  # [384, 13, 13]
            nn.ReLU(inplace=True),

            # C5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  # [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [256, 6, 6]
        )
        self.flatten = nn.Flatten()     # 将张量压平到一维
        self.classifier = nn.Sequential(
            # Dropout 过大容易欠拟合，Dropout 过小速度慢、或容易过拟合
            nn.Dropout(p=0.5),  # inplace 操作会导致梯度计算所需的变量被修改
            nn.Linear(in_features=256 * 6 * 6, out_features=num_features),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=num_features, out_features=num_features),
            nn.ReLU(),

            nn.Linear(in_features=num_features, out_features=num_classes),
        )

    def forward(self, x):
        x = self.net(x)  # 卷积层，特征提取
        x = self.flatten(x)  # 将张量维度压平到一维，以便全连接层计算
        x = self.classifier(x)  # 全连接层以及分类
        return x


class AlexNet_CIFAR10(nn.Module):
    """
    用于测试模型的轻量化版本，减少模型参数量
    """

    def __init__(self, num_features=None, num_classes=10):
        """
        图像的输入形状为 [3, 32, 32]
        Args:
            num_features: 全连接层的输入维度，如果不指定，则默认为 16 * num_classes
            num_classes: 分类个数
        """
        if num_features is None:
            num_features = 16 * num_classes
        super(AlexNet_CIFAR10, self).__init__()
        self.net = nn.Sequential(
            # C1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # C2
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # C3
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # C4
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # C5
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128 * 1 * 1, out_features=num_features),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=num_features, out_features=num_features),
            nn.ReLU(),

            nn.Linear(in_features=num_features, out_features=num_classes),
        )

    def forward(self, x):
        x = self.net(x)  # 卷积层，特征提取
        x = self.flatten(x)  # 将张量维度压平到一维，以便全连接层计算
        x = self.classifier(x)  # 全连接层以及分类
        return x


if __name__ == '__main__':
    _model = AlexNet_CIFAR10(40, 10)
    x_input = torch.randn(1, 3, 32, 32)
    y_pred = _model(x_input)
    print(y_pred.shape)

    log_model_params(_model, input_size=x_input.shape)
