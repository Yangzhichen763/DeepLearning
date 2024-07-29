import pywt     # 全称是 PyWavelets
import pywt.data
import torch
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    """
    create wavelet filters for convolutional layers
    创建卷积层的小波滤波器
    """
    def create_filters(high, low):
        filters = torch.stack([
            low.unsqueeze(0) * low.unsqueeze(1),
            low.unsqueeze(0) * high.unsqueeze(1),
            high.unsqueeze(0) * low.unsqueeze(1),
            high.unsqueeze(0) * high.unsqueeze(1)], dim=0)
        return filters

    w = pywt.Wavelet(wave)

    # Decomposition filters 分解滤波
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)    # Highpass decomposition filter
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)    # Lowpass decomposition filter
    dec_filters = create_filters(dec_hi, dec_lo)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # Reconstruction filters 重构滤波
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])   # Highpass reconstruction filter
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])   # Lowpass reconstruction filter
    rec_filters = create_filters(rec_hi, rec_lo)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


# noinspection PyPep8Naming
def wavelet_transform(x, filters):
    """
    wavelet transform using convolutions
    小波变换
    """
    B, C, H, W = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=C, padding=pad)
    x = x.reshape(B, C, 4, H // 2, W // 2)
    return x


# noinspection PyPep8Naming
def inverse_wavelet_transform(x, filters):
    """
    inverse wavelet transform using convolutions
    小波逆变换
    """
    B, C, _, H_half, W_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(B, C * 4, H_half, W_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=C, padding=pad)
    return x
