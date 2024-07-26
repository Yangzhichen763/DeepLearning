import torch
import torch.nn as nn


"""
Global Filter，使用 fft 实现在频域中学习空间位置的相互作用，可以捕获全局信息，不会引入归纳偏差
    - 对数线性时间复杂度O(NlogN)，减少了学习成本和计算量；
    - 全局滤波层，相当于尺寸为 HxW 的 depthwise 全局循环卷积滤波器；
    - GFNet 更偏向于在频域捕捉关系；
    - 实验中表明，模型的全局滤波器比局部卷积层更好。
论文链接 2021：https://arxiv.org/abs/2107.00645
"""


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation=nn.GELU, dropout=0.0):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation(),
            self.dropout,
            nn.Linear(hidden_channels, out_channels),
            self.dropout
        )

    def forward(self, x):
        return self.net(x)


class GlobalFilter(nn.Module):
    """
    代码修改自：https://github.com/raoyongming/GFNet/blob/master/gfnet.py
    先将图像 embedding 后再传入
    """
    def __init__(self, channels, patch_size=16):
        super(GlobalFilter, self).__init__()
        self.patch_size = patch_size
        self.height = patch_size
        self.width = patch_size // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(self.height, self.width, channels, 2, dtype=torch.float32) * 0.02)

    # noinspection PyPep8Naming
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape: [B, N, C]
        """
        dtype = x.dtype
        x = x.float()

        B, N, C = x.shape
        P = self.patch_size

        assert N == P * P, "输入 x 的尺寸必须为 [B, P*P, C]"

        x = x.view(B, P, P, C)                                       # [B, N, C] -> [B, P, P, C]
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')             # [B, P, P, C] -> [B, H, W, C]; H=P, W=P//2+1
        weight = torch.view_as_complex(self.complex_weight)          # [H, W, C, 2] -> [H, W, C]
        x = x * weight
        x = torch.fft.irfft2(x, s=(P, P), dim=(1, 2), norm='ortho')  # [B, H, W, C] -> [B, P, P, C]

        x = x.reshape(B, N, C)                                       # [B, P, P, C] -> [B, N, C]
        x = x.type(dtype)

        return x


class GlobalFilterBlock(nn.Module):
    def __init__(self, channels, mlp_hidden_dim, patch_size=16, activation=nn.GELU, norm=nn.LayerNorm, dropout=0.0):
        """
        Args:
            channels (int): 输入通道数
            mlp_hidden_dim (int): MLP 隐藏层维度
            patch_size (int): 输入图像 patch 大小
            activation: 激活函数
            norm: 归一化层
            dropout (float): dropout 率
        """
        super(GlobalFilterBlock, self).__init__()
        self.net = nn.Sequential(
            norm(channels),
            GlobalFilter(channels, patch_size),
            norm(channels),
            MLP(channels, mlp_hidden_dim, channels, activation=activation, dropout=dropout)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    from utils.log.model import log_model_params

    _patch_size, _channels = 16, 3
    x_input = torch.randn(2, _patch_size * _patch_size, _channels)
    model = GlobalFilterBlock(_channels, 128, _patch_size)
    log_model_params(model, input_data=x_input)
