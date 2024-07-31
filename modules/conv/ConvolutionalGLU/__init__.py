
"""
TransNeXt: Robust Foveal Visual Perception for Vision Transformers

论文链接 2023：https://arxiv.org/abs/2311.17132
"""


import torch
import torch.nn as nn


"""
Convolutional Gated-Linear-Unit(Convolutional GLU):
  - 门控通道注意力机制门控线性单元（GLU）是一种通道混频器，已被证明再各种自然语言处理任务中性能优于多层感知器（MLP）
  - 在 GLU 的门控分支的激活函数之前添加一个最小形式的 3x3 深度卷积，就可以将其转化为基于最近邻特征的门控通道注意力机制

参考代码：https://github.com/DaiShiResearch/TransNeXt/blob/main/classification/transnext.py
"""


# 该模块是论文中使用到的，而非用于对比的
class DWConv(nn.Module):
    """
    Depthwise convolution
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    # noinspection PyPep8Naming
    def forward(self, x, H, W):
        """
        Args:
            x (torch.Tensor): [B, N, C]
            H (int): height of feature map
            W (int): width of feature map
        """
        B, N, C = x.shape

        assert N % (H * W) == 0, f"N should be divisible by H * W, instead of {N} % {H * W} = {N % (H * W)}"

        x = x.transpose(1, 2).view(B, C, H, W).contiguous()  # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        x = self.dw_conv(x)                                              # [B, C, H, W] -> [B, C, H, W]
        x = x.flatten(start_dim=2, end_dim=-1).transpose(1, 2)           # [B, C, H, W] -> [B, C, N] -> [B, N, C]

        return x


class SEModule(nn.Module):
    """
    更泛的通道注意力机制见 CBAM(Convolutional Block Attention Module)
    """
    def __init__(self, in_channels, squeeze_factor=16):
        super(SEModule, self).__init__()

        assert in_channels // squeeze_factor > 0, \
            (f"Squeeze factor should be less than or equal to the number of input channels,"
             f" instead of {in_channels} // {squeeze_factor} = {in_channels // squeeze_factor}")
        assert in_channels % squeeze_factor == 0, \
            (f"Squeeze factor should be divisible by the number of input channels,"
             f" instead of {in_channels} % {squeeze_factor} = {in_channels % squeeze_factor}")

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // squeeze_factor)
        self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // squeeze_factor, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, N, C]
        """
        y = self.avg_pool(x)                   # [B, N, C] -> [B, N, 1]
        y = self.fc1(y.squeeze(dim=-1))        # [B, N, 1] -> [B, N] -> [B, dim], where dim = C // squeeze_factor
        y = self.activation(y)
        y = self.fc2(y)                        # [B, dim] -> [B, C]
        y = self.sigmoid(y).unsqueeze(dim=-1)  # [B, C] -> [B, C, 1]
        return x * y


class OriginalFeedForwardLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.in_fc = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.out_fc = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # noinspection PyPep8Naming
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, N, C]
        """
        x = self.in_fc(x)   # [B, N, Cin] -> [B, N, dim]
        x = self.act(x)
        x = self.drop(x)
        x = self.out_fc(x)  # [B, N, dim] -> [B, N, Cout]
        x = self.drop(x)
        return x


class ConvolutionalFeedForwardLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.in_fc = nn.Linear(in_features, hidden_features)
        self.dw_conv = DWConv(hidden_features)
        self.act = act_layer()
        self.out_fc = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # noinspection PyPep8Naming
    def forward(self, x, H=28, W=28):
        """
        Args:
            x (torch.Tensor): [B, N, C]
            H (int): height of feature map
            W (int): width of feature map
        """
        x = self.in_fc(x)                    # [B, N, Cin] -> [B, N, dim]
        x = self.act(self.dw_conv(x, H, W))
        x = self.drop(x)
        x = self.out_fc(x)                   # [B, N, dim] -> [B, N, Cout]
        x = self.drop(x)
        return x


class GLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.in_fc = nn.Linear(in_features, hidden_features * 2)
        self.act = act_layer()
        self.out_fc = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # noinspection PyPep8Naming
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, N, C]
        """
        x, v = self.in_fc(x).chunk(2, dim=-1)  # [B, N, Cin] -> 2*[B, N, dim]
        x = self.act(x) * v
        x = self.drop(x)
        x = self.out_fc(x)                     # [B, N, dim] -> [B, N, Cout]
        x = self.drop(x)
        return x


class SEFeedForwardLayer(nn.Module):
    def __init__(self, in_num, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.in_fc = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.out_fc = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.se = SEModule(in_num)

    # noinspection PyPep8Naming
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, N, C]
        """
        x = self.in_fc(x)   # [B, N, Cin] -> [B, N, dim]
        x = self.act(x)
        x = self.drop(x)
        x = self.out_fc(x)  # [B, N, dim] -> [B, N, Cout]
        x = self.drop(x)
        x = self.se(x)
        return x


# 该模块是论文中使用到的，而非用于对比的
class ConvolutionalGLU(nn.Module):
    """
    Convolutional Gated-Linear-Unit(GLU)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.in_fc = nn.Linear(in_features, hidden_features * 2)
        self.dw_conv = DWConv(hidden_features)
        self.act = act_layer()
        self.out_fc = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # noinspection PyPep8Naming
    def forward(self, x, H=28, W=28):
        """
        Args:
            x (torch.Tensor): [B, N, C]
            H (int): height of feature map
            W (int): width of feature map
        """
        x, v = self.in_fc(x).chunk(2, dim=-1)    # [B, N, Cin] -> 2*[B, N, dim]
        x = self.act(self.dw_conv(x, H, W)) * v
        x = self.drop(x)
        x = self.out_fc(x)                       # [B, N, dim] -> [B, N, Cout]
        x = self.drop(x)
        return x


# noinspection PyPep8Naming
if __name__ == '__main__':
    from utils.log.model import log_model_params

    x_input = torch.randn(2, 784, 32)
    _N, _C = x_input.shape[1], x_input.shape[2]

    # test SEModule
    print("  > Test SEModule")
    model = SEModule(_N)
    log_model_params(model, input_data=x_input)

    # test OriginalFeedForwardLayer
    print("  > Test OriginalFeedForwardLayer")
    model = OriginalFeedForwardLayer(_C, _C, _C, nn.GELU, 0.1)
    log_model_params(model, input_data=x_input)

    # test _ConvolutionalFeedForwardLayer
    print("  > Test ConvolutionalFeedForwardLayer")
    model = ConvolutionalFeedForwardLayer(_C, _C, _C, nn.GELU, 0.1)
    log_model_params(model, input_data=x_input)

    # test GLU
    print("  > Test GLU")
    model = GLU(_C, _C, _C, nn.GELU, 0.1)
    log_model_params(model, input_data=x_input)

    # test SEFeedForwardLayer
    print("  > Test SEFeedForwardLayer")
    model = SEFeedForwardLayer(_N, _C, _C, _C, nn.GELU, 0.1)
    log_model_params(model, input_data=x_input)

    # test ConvolutionalGLU
    print("  > Test ConvolutionalGLU")
    model = ConvolutionalGLU(_C, _C, _C, nn.GELU, 0.1)
    log_model_params(model, input_data=x_input)




