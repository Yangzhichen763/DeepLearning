import torch
import torch.nn as nn

from timm.models.layers import DropPath

# noinspection SpellCheckingInspection
"""
TSLANet: Time Series Lightweight Adaptive Network
  - 用于处理时间序列预测（TSLANet_Forecasting）或者分类（TSLANet_Classification）问题
  
论文链接 2024：https://arxiv.org/abs/2404.08472
参考代码：https://github.com/emadeldeen24/TSLANet/blob/main/Classification/TSLANet_classification.py
  - 该代码也是很好的学习 lightning, timm 等第三方库的代码
  - 论文和代码都可以参考，是值得学习的资料
  
@inproceedings{tslanet,
  title     = {TSLANet: Rethinking Transformers for Time Series Representation Learning},
  author    = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Li, Xiaoli},
  booktitle = {International Conference on Machine Learning},
  year      = {2024},
}
"""


class InteractiveConvBlock1d(nn.Module):
    """
    Interactive Convolutional Block (ICB) for 1D inputs
    ICB 通过不同核大小的并行卷积，可以捕获局部特征和较远距离的依赖关系
    """
    def __init__(self, in_channels, hidden_dim, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, N, C)
        """
        x = x.transpose(-2, -1)     # [B, N, C] -> [B, C, N]

        x1 = self.conv1(x)
        x1_act = self.activation(x1)
        x1_drop = self.dropout(x1_act)

        x2 = self.conv2(x)
        x2_act = self.activation(x2)
        x2_drop = self.dropout(x2_act)

        out1 = x1 * x2_drop
        out2 = x2 * x1_drop

        x = self.conv3(out1 + out2)
        x = x.transpose(-2, -1)     # [B, C, N] -> [B, N, C]
        return x


class InteractiveConvBlock2d(nn.Module):
    """
    Interactive Convolutional Block (ICB) for 2D inputs
    """
    def __init__(self, in_channels, hidden_dim, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, C, H, W)
        """
        x1 = self.conv1(x)
        x1_act = self.activation(x1)
        x1_drop = self.dropout(x1_act)

        x2 = self.conv2(x)
        x2_act = self.activation(x2)
        x2_drop = self.dropout(x2_act)

        out1 = x1 * x2_drop
        out2 = x2 * x1_drop

        x = self.conv3(out1 + out2)
        return x


class AdaptiveSpectralBlock1d(nn.Module):
    """
    Adaptive Spectral Block (ASB) for 1D inputs
    其中包含一个自适应局部滤波器，允许模型根据数据集特征动态调整滤波水平，并去除这些高频噪声成分。
       - 该滤波器自适应地为每个特定的时间序列数据设置合适的频率阈值，可以很好地处理频谱可能随时间变化的非平稳数据
       - 高频分量通常代表偏离潜在趋势或信号的快速波动，使它们看起来更加随机且难以解释
    """
    def __init__(self, channels):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(channels, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(channels, 2, dtype=torch.float32) * 0.02)

        torch.nn.init.trunc_normal_(self.complex_weight_high, mean=0.0, std=0.02)
        torch.nn.init.trunc_normal_(self.complex_weight, mean=0.0, std=0.02)

        self.threshold_param = nn.Parameter(torch.rand(1, dtype=torch.float32))

    # noinspection PyPep8Naming
    def create_adaptive_high_freq_mask(self, x_fft: torch.Tensor):
        B = x_fft.shape[0]

        # Calculate energy in the frequency domain
        # [B, N, C] -> [B, N]
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        # [B, N] -> [B, 1]
        median_energy = energy.median(dim=-1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)  # [B, 1] -> [B, 1, 1]

        return adaptive_mask

    # noinspection PyPep8Naming
    def forward(self, x: torch.Tensor, adaptive_filter=True):
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, N, C)
            adaptive_filter (bool): whether to apply adaptive filter or not
        """
        dtype = x.dtype
        x = x.to(torch.float32)

        B, N, C = x.shape

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')          # [B, N, C] -> [B, N', C]; N'=N//2+1
        weight = torch.view_as_complex(self.complex_weight)     # [C, 2] -> [C]
        x_weighted = x_fft * weight                             # [B, N', C] * [C] -> [B, N', C]

        if adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)  # [B, 1, 1]
            x_masked = x_fft * freq_mask.to(x.device)               # [B, N', C] * [B, 1, 1] -> [B, N', C]

            weight_high = torch.view_as_complex(self.complex_weight_high)    # [C, 2] -> [C]
            x_weighted2 = x_masked * weight_high                             # [B, N', C] * [C] -> [B, N', C]

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')    # [B, N', C] -> [B, N, C]

        x = x.reshape(B, N, C)  # Reshape back to original shape
        x = x.to(dtype)

        return x


class AdaptiveSpectralBlock2d(nn.Module):
    """
    Adaptive Spectral Block (ASB) for 2D inputs
    """
    def __init__(self, channels, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.height = patch_size
        self.width = patch_size // 2 + 1

        self.complex_weight_high = nn.Parameter(torch.randn(self.height, self.width, channels, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(self.height, self.width, channels, 2, dtype=torch.float32) * 0.02)

        torch.nn.init.trunc_normal_(self.complex_weight_high, mean=0.0, std=0.02)
        torch.nn.init.trunc_normal_(self.complex_weight, mean=0.0, std=0.02)

        self.threshold_param = nn.Parameter(torch.rand(1, dtype=torch.float32))

    # noinspection PyPep8Naming
    def create_adaptive_high_freq_mask(self, x_fft: torch.Tensor):
        B = x_fft.shape[0]

        # Calculate energy in the frequency domain
        # [B, H, W, C] -> [B, H, W]
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        # [B, H, W] -> [B, H*W] -> [B, 1]
        flat_energy: torch.Tensor = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=-1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)  # [B, 1, 1] -> [B, 1, 1, 1]

        return adaptive_mask

    # noinspection PyPep8Naming
    def forward(self, x: torch.Tensor, adaptive_filter=True):
        """
        以处理 patch 的方式进行处理，所以输入还是 (B, N, C)

        Args:
            x (torch.Tensor): input tensor with shape (B, N, C)
            adaptive_filter (bool): whether to apply adaptive filter or not
        """
        dtype = x.dtype
        x = x.to(torch.float32)

        B, N, C = x.shape
        P = self.patch_size

        assert N == P * P, "输入 x 的尺寸必须为 [B, P*P, C]"

        x = x.view(B, P, P, C)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')    # [B, P, P, C] -> [B, H, W, C]; H=P, W=P//2+1
        weight = torch.view_as_complex(self.complex_weight)     # [H, W, C, 2] -> [H, W, C]
        x_weighted = x_fft * weight                             # [B, H, W, C] * [H, W, C] -> [B, H, W, C]

        if adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)  # [B, 1, 1, 1]
            x_masked = x_fft * freq_mask.to(x.device)               # [B, H, W, C] * [B, 1, 1, 1] -> [B, H, W, C]

            weight_high = torch.view_as_complex(self.complex_weight_high)    # [H, W, C, 2] -> [H, W, C]
            x_weighted2 = x_masked * weight_high                             # [B, H, W, C] * [H, W, C] -> [B, H, W, C]

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft2(x_weighted, s=(P, P), dim=(1, 2), norm='ortho')    # [B, H, W, C] -> [B, P, P, C]

        x = x.reshape(B, N, C)  # Reshape back to original shape
        x = x.to(dtype)

        return x


class TSLALayer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., dropout=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = AdaptiveSpectralBlock2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = InteractiveConvBlock1d(in_channels=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x_residual = x
        x = self.norm1(x)
        x = self.asb(x)
        x = self.norm2(x)
        x = self.icb(x)
        x = self.drop_path(x)
        return x + x_residual


if __name__ == '__main__':
    from utils.log.model import log_model_params

    # 测试 AdaptiveSpectralBlock1d
    # x_input = torch.randn(2, 16, 12)
    # model = AdaptiveSpectralBlock1d(12)

    # 测试 AdaptiveSpectralBlock2d
    # x_input = torch.randn(2, 256, 3)
    # model = AdaptiveSpectralBlock2d(3, 16)

    # 测试 TSLALayer
    x_input = torch.randn(2, 256, 3)
    model = TSLALayer(3)

    log_model_params(model, input_data=x_input)