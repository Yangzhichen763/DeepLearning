import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Adaptive Fourier Neural Operator (AFNO)
    - 对复数权重分块；
    - 在下游任务和 ImageNet 识别任务中超越以往傅里叶变换的 token mixer 模型。
论文链接 2021：https://arxiv.org/abs/2111.13587
"""


class AFNO2D(nn.Module):
    """
    代码修改自：https://github.com/lonestar686/AdaptiveFourierNeuralOperator/blob/master/afno/afno2d.py
    """
    def __init__(self, channels, patch_size=16,
                 num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        """
        Args:
            channels: 输入输出的通道数
            patch_size:
            num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
            sparsity_threshold: lambda for softshrink
            hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
            hidden_size_factor: 隐藏层的通道数的倍数
        """
        super().__init__()
        assert channels % num_blocks == 0, f"hidden_size {channels} should be divisble by num_blocks {num_blocks}"

        self.patch_size = patch_size

        self.hidden_dim = channels
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_dim // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    # noinspection PyPep8Naming
    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()

        B, N, C = x.shape
        P = self.patch_size
        H = P
        W = P // 2 + 1
        # nb = self.num_blocks
        # bs = self.block_size

        assert N == P * P, "输入 x 的尺寸必须为 [B, P*P, C]"

        # 傅里叶变换
        x = x.reshape(B, P, P, C)                                 # [B, N, C] -> [B, P, P, C]
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")          # [B, P, P, C] -> [B, H, W, C]; H=P, W=P//2+1
        x = x.reshape(B, H, W, self.num_blocks, self.block_size)  # [B, H, W, C] -> [B, H, W, nb, bs]

        # 计算虚数卷积
        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) +
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) +
            self.b2[1]
        )

        # 逆傅里叶变换
        x = torch.stack([o2_real, o2_imag], dim=-1)           # [B, H, W, nb, bs] -> [B, H, W, nb, bs, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)                                 # [B, H, W, nb, bs, 2] -> [B, H, W, nb, bs]
        x = x.reshape(B, x.shape[1], x.shape[2], C)           # [B, H, W, nb, bs] -> [B, H, W, C]
        x = torch.fft.irfft2(x, s=(P, P), dim=(1, 2), norm="ortho")  # [B, H, W, C] -> [B, P, P, C]

        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias


if __name__ == '__main__':
    from utils.log.model import log_model_params

    _patch_size, _channels = 16, 128
    x_input = torch.randn(2, _patch_size * _patch_size, _channels)
    model = AFNO2D(channels=_channels, patch_size=_patch_size)
    y_output = model(x_input)

    log_model_params(model, input_data=x_input)
    print(y_output.shape)
