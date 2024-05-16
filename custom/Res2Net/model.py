import torch
import torch.nn as nn
import math

from torchsummary import summary

class Conv(nn.Module):  # 定义一个卷积模块类
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 act=True):
        """ 构造函数
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 组数
            act: 是否使用激活函数
        """
        super(Conv, self).__init__()  # 调用父类构造函数
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=False)  # 定义卷积层
        self.bn = nn.BatchNorm2d(out_channels)  # 定义BN层
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()  # 定义激活函数

    def forward(self, x):
        x = self.conv(x)  # 卷积操作
        x = self.bn(x)  # BN操作
        x = self.act(x)  # 激活函数
        return x  # 返回输出

class Res2NetBottle2neck(nn.Module):  # 定义一个Res2Net的瓶颈模块类
    expansion = 1  # 用于确定后续通道数的扩展比例，默认为1

    def __init__(self, in_planes, planes, shortcut, baseWidth=26, scale=4):
        """ 构造函数
        Args:
            in_planes: 输入通道维度
            planes: 输出通道维度
            shortcut: 是否使用shortcut连接
            baseWidth: conv3x3的基本宽度
            scale: 尺度的数量
        """
        super(Res2NetBottle2neck, self).__init__()  # 调用父类构造函数
        width = int(math.floor(planes * (baseWidth / 64.0)))  # 计算宽度
        self.conv1 = Conv(in_planes, width * scale, kernel_size=1, padding=0)  # 第一个1x1卷积操作
        if scale == 1:  # 如果尺度为1
            self.nums = 1
        else:
            self.nums = scale - 1  # 计算尺度数
        convs = []  # 初始化卷积列表
        for i in range(self.nums):  # 针对每个尺度
            convs.append(Conv(width, width, kernel_size=3))  # 添加一个3x3卷积操作到列表中
        self.convs = nn.ModuleList(convs)  # 将卷积列表转换为ModuleList类型
        print("planes", planes, "expansion", self.expansion)
        self.conv3 = Conv(width * scale, planes * self.expansion, kernel_size=1, padding=0, act=False)  # 最后一个1x1卷积操作
        self.silu = nn.SiLU(inplace=True)  # 激活函数
        self.scale = scale  # 保存尺度
        self.width = width  # 保存宽度
        self.shortcut = shortcut  # 是否使用shortcut连接的标志

    def forward(self, x):
        if self.shortcut:  # 如果使用shortcut连接。（这一步骤是为了能和原本C3当中的参数对应上）
            residual = x  # 保存残差

        out = self.conv1(x)  # 第一个1x1卷积操作，降维
        spx = torch.split(out, self.width, 1)  # 在通道维度上分割张量
        print("spx len", len(spx))
        for i in range(self.nums):  # 对于每个尺度
            if i == 0:
                sp = spx[i]  # 如果是第一个尺度，取出分割后的张量
            else:
                sp = sp + spx[i]  # 否则，累加之前的张量和当前尺度的张量
            sp = self.convs[i](sp)  # 进行3x3卷积操作
            print("sp", sp.shape)
            if i == 0:
                out = sp  # 如果是第一个尺度，直接赋值给out
            else:
                out = torch.cat((out, sp), 1)  # 否则，将当前尺度的结果和之前的结果在通道维度上拼接
        if self.scale != 1:  # 如果尺度不为1
            print("out", out.shape)
            print("self.nums", self.nums, "spx[self.nums]", spx[self.nums].shape)
            out = torch.cat((out, spx[self.nums]), 1)  # 在通道维度上将之前的结果和剩余的通道拼接
        out = self.conv3(out)  # 最后一个1x1卷积操作
        print("out", out.shape)

        if self.shortcut:  # 如果使用shortcut连接
            print("residual", residual.shape)
            out += residual  # 加上残差

        out = self.silu(out)  # 使用SiLU激活函数
        return out  # 返回输出


if __name__ == '__main__':
    model = Res2NetBottle2neck(32, 32, True)
    x_input = torch.randn(1, 32, 224, 224)
    y_pred = model(x_input)
    print(y_pred.shape)

    input_shape = x_input.shape;
    device = torch.device("cuda")
    model = model.to(device)
    summary(model=model,
            input_size=tuple(input_shape)[1:],  # tuple(input_shape) 是 (1, 3, 224, 224)
            # batch_size=input_shape[0],
            device=device.__str__())

