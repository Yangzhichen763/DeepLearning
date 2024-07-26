import torch.nn as nn


"""
Depthwise Separable Convolution 在 MobileNetV2 中被提及，旨在减少模型参数量和计算量，从而提高模型的效率和性能
  - 传统的卷积操作在每个输入通道上使用一个卷积核来进行卷积操作，而深度可分离卷积将这个操作拆分成深度卷积和逐点卷积。
  - 深度卷积：在每个输入通道上使用一个卷积核，不改变输入数据的通道数
  - 逐点卷积：在每个像素点上使用一个卷积核，卷积核大小为 1×1xC
    
论文链接：
    MobileNet-v1 2017: https://arxiv.org/abs/1704.04861
        - 旨在设计适用于移动设备和嵌入式设备的轻量级卷积神经网络结构。
    MobileNet-v2 2018: https://arxiv.org/abs/1801.04381
        - 引入了倒残差结构和线性瓶颈，提高了模型的性能和效率。
    MobileNet-v3 2019: https://arxiv.org/abs/1905.02244
        - 进一步优化了网络结构，引入了可调节的网络宽度和分辨率以提高模型的灵活性和性能。
"""


class DepthwiseConv2d(nn.Module):
    """
    深度卷积 (Depthwise Convolution, DWConv)，深度卷积中一个卷积核只有一维，负责一个通道，一个通道只被一个卷积核卷积
    """
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class PointwiseConv2d(nn.Module):
    """
    逐点卷积 (Pointwise Convolution, PWConv)，逐点卷积卷积核大小为 1×1xC（C为输入数据的维度），每次卷积一个像素的区域。
    """
    def __init__(self, in_channels, out_channels, bias=False):
        super(PointwiseConv2d, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0,
                                   dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    """
    深度可分离卷积 (Depthwise Separable Convolution)，由深度卷积和逐点卷积组成
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.net = nn.Sequential(
            DepthwiseConv2d(in_channels, kernel_size, stride, padding, dilation, bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            PointwiseConv2d(in_channels, out_channels, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


if __name__ == '__main__':
    import torch
    from utils.log.model import log_model_params

    x_input = torch.randn(1, 32, 224, 224)
    model = DepthwiseSeparableConv2d(32, 64, 3, padding=1)
    log_model_params(model, input_data=x_input)
