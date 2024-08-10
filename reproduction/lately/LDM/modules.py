import enum
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from modules.attention import SpatialSelfAttention, LinearAttention
from modules.activation import Swish
from modules.embedding.timestep import get_timestep_embedding
from modules.residual import ShortCutType


"""
参考代码：https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
"""


class AttentionType(enum.Enum):
    NONE = enum.auto()
    VANILLA = enum.auto()
    LINEAR = enum.auto()

    def make_attn(self, in_dim):
        if self == AttentionType.NONE:
            return None
        elif self == AttentionType.VANILLA:
            return SpatialSelfAttention(in_dim)
        elif self == AttentionType.LINEAR:
            return LinearAttention(in_dim)
        else:
            raise ValueError(f"Unknown attention type: {self}")


class TimeEmbeddingProjection(nn.Module):
    """
    Time embedding projection module.
    """
    # noinspection PyPep8Naming
    def __init__(self, in_dim, embedding_dim):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            Swish(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        embedding = get_timestep_embedding(t, self.embedding_dim)
        embedding = self.time_embedding(embedding)
        return embedding


# 在下面几个模块中使用到的 norm 层
class Norm(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.norm(x)


# 单个上采样（插值块）卷积块，用于特征上采样
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


# 单个下采样卷积块（或者池化块），用于特征下采样
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:  # 添加非对称 padding
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


# 紧跟着采样块的残差卷积模块
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, time_embedding_channels=512,
                 short_cut=ShortCutType.IDENTITY, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        self.out_channels = out_channels
        self.short_cut = short_cut

        self.conv1 = nn.Sequential(
            Norm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        if time_embedding_channels > 0:
            self.t_embedding_projection = torch.nn.Linear(time_embedding_channels, out_channels)

        self.conv2 = nn.Sequential(
            Norm(out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # residual connection
        if self.short_cut == ShortCutType.IDENTITY:
            self.short_cut = nn.Identity()
        elif self.short_cut == ShortCutType.CONV1X1:
            self.short_cut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        elif self.short_cut == ShortCutType.CONV3X3:
            self.short_cut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unknown short cut type: {self.short_cut}")

    def forward(self, x, t_embedding):
        h = x

        h = self.conv1(h)
        if t_embedding is not None:
            h = h + self.t_embedding_projection(Swish(t_embedding))[:, :, None, None]
        h = self.conv2(h)

        short_cut = self.short_cut(x)
        return short_cut + h


# time-embedding-based UNet
class Model(nn.Module):
    def __init__(self, *,
                 in_channels, out_channels, feature_dims=(128, 256, 512, 1024),             # 通道数
                 num_res_blocks, resamp_with_conv=True,                                     # 采样模块参数
                 resolution, attention_resolutions, attention_type=AttentionType.VANILLA,   # 注意力模块参数
                 use_timestep=True,                                                         # 时间步长嵌入参数
                 dropout=0.0):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            feature_dims (tuple[int] or list[int]): 各层特征图的通道数
            num_res_blocks: 每层的残差块数
            resamp_with_conv (bool): residual-sampling with convolution 是否使用卷积进行特征上采样
            resolution: 输入图像或特征图的分辨率
            attention_resolutions (tuple[int] or list[int]): 每层注意力模块输入特征图的分辨率
            attention_type: 注意力模块的类型
            use_timestep: 是否使用时间步长嵌入
            dropout: 卷积层的 dropout 率
        """
        super().__init__()
        self.channels = feature_dims[0]
        self.t_embedding_channels = self.channels * 4
        self.num_resolutions = len(feature_dims)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        in_dim = self.channels

        # timestep embedding
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.t_embedding = TimeEmbeddingProjection(self.channels, self.t_embedding_channels)

        # down-sampling
        # 通道转换，由输入通道转换为第一个卷积层的输入通道数
        self.conv_in = torch.nn.Conv2d(in_channels, self.channels, kernel_size=3, stride=1, padding=1)
        # 下采样模块的堆叠
        current_resolution = resolution                                 # 当前的采样块的输入分辨率
        in_feature_dims = (self.channels,) + tuple(feature_dims)[:-1]   # 单个采样块的输入通道数
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            in_dim, out_dim = in_feature_dims[i_level], feature_dims[i_level]   # 采样块的输入输出通道数
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(                                        # 卷积模块
                        in_channels=in_dim,
                        out_channels=out_dim,
                        time_embedding_channels=self.t_embedding_channels,
                        dropout=dropout
                    ))
                in_dim = out_dim
                if current_resolution in attention_resolutions:
                    attn.append(attention_type.make_attn(in_dim))       # 注意力模块

            # 整个下采样模块：卷积->注意力->下采样
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:  # 最后一个下采样模块不需要下采样
                down.downsample = Downsample(in_dim, resamp_with_conv)  # 采样模块
                current_resolution = current_resolution // 2
            self.down.append(down)

        # middle
        # UNet 中间模块：卷积->注意力->卷积
        self.mid = nn.Module()
        self.mid.block_in = ResnetBlock(
            in_channels=in_dim,
            out_channels=in_dim,
            time_embedding_channels=self.t_embedding_channels,
            dropout=dropout)
        self.mid.attention = attention_type.make_attn(in_dim)
        self.mid.block_out = ResnetBlock(
            in_channels=in_dim,
            out_channels=in_dim,
            time_embedding_channels=self.t_embedding_channels,
            dropout=dropout)

        # up-sampling
        # 上采样模块的堆叠
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            out_dim, skip_in = feature_dims[i_level], feature_dims[i_level]  # 采样块的输入和跳跃链接的通道数
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = in_channels * in_feature_dims[i_level]
                block.append(
                    ResnetBlock(                                        # 卷积模块
                        in_channels=in_dim + skip_in,
                        out_channels=out_dim,
                        time_embedding_channels=self.t_embedding_channels,
                        dropout=dropout
                    ))
                in_dim = out_dim
                if current_resolution in attention_resolutions:
                    attn.append(attention_type.make_attn(in_dim))       # 注意力模块

            # 整个上采样模块：卷积->注意力->上采样
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(in_dim, resamp_with_conv)        # 采样模块
                current_resolution = current_resolution * 2
            self.up.insert(0, up)   # 与下采样的顺序相反

        # end
        # 输出层
        self.conv_out = nn.Conv2d(in_dim, out_channels, kernel_size=3, stride=1, padding=1)  # 用于需要获取最后一层的权重
        self.net_out = nn.Sequential(
            Norm(in_dim),
            Swish(),
            self.conv_out
        )

    def forward(self, x, t=None, context=None):
        # assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)

        # timestep embedding
        if self.use_timestep:
            assert t is not None
            time_embedding = self.t_embedding(t)
        else:
            time_embedding = None

        # down-sampling
        # ((卷积->注意力)*blocks ->下采样)*layers
        h_skip = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):     # 逐层处理，(卷积->注意力)*n ->下采样
            for i_block in range(self.num_res_blocks):  # 逐块处理，卷积->注意力
                h = self.down[i_level].block[i_block](h_skip[-1], time_embedding)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h_skip.append(h)
            # 记录 skip 连接
            if i_level != self.num_resolutions - 1:     # 下采样，最后一个下采样模块不需要记录到 skip 中
                h_skip.append(self.down[i_level].downsample(h_skip[-1]))

        # middle
        # 卷积->注意力->卷积
        h = h_skip[-1]
        h = self.mid.block_in(h, time_embedding)
        h = self.mid.attention(h)
        h = self.mid.block_out(h, time_embedding)

        # up-sampling
        # ((卷积->注意力)*blocks ->上采样)*layers
        for i_level in reversed(range(self.num_resolutions)):  # 逐层处理，(卷积->注意力)*n ->上采样
            for i_block in range(self.num_res_blocks + 1):     # 逐块处理，卷积->注意力
                h = self.up[i_level].block[i_block](
                    torch.cat([h, h_skip.pop()], dim=1), time_embedding)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # 记录 skip 连接
            if i_level != 0:                                   # 上采样，第一个上采样模块不需要记录到 skip 中
                h = self.up[i_level].upsample(h)

        # end
        h = self.net_out(h)

        return h

    def get_last_layer(self):
        return self.conv_out.weight


# time-embedding-based Encoder，输入图像通道数为 in_channels，输出特征图通道数为 z_channels
class Encoder(nn.Module):
    def __init__(self, *,
                 in_channels, z_channels, any_double_z=True, feature_dims=(128, 256, 512, 1024),
                 num_res_blocks, resamp_with_conv=True,
                 resolution, attention_resolutions, attention_type=AttentionType.VANILLA,
                 dropout=0.0):
        """
        Args:
            in_channels: 输入通道数
            z_channels: 编码器输出的通道数，也就是特征图的通道数
            any_double_z: 是否将编码器输出的通道数翻倍
            feature_dims (tuple[int] or list[int]): 各层特征图的通道数
            num_res_blocks: 每层的残差块数
            resamp_with_conv (bool): residual-sampling with convolution 是否使用卷积进行特征上采样
            resolution: 输入图像或特征图的分辨率
            attention_resolutions (tuple[int] or list[int]): 每层注意力模块输入特征图的分辨率
            attention_type: 注意力模块的类型
            dropout: 卷积层的 dropout 率
        """
        super().__init__()
        self.channels = feature_dims[0]
        self.t_embedding_channels = self.channels * 4
        self.num_resolutions = len(feature_dims)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        in_dim = self.channels

        # down-sampling
        # 通道转换，由输入通道转换为第一个卷积层的输入通道数
        self.conv_in = torch.nn.Conv2d(in_channels, in_dim, kernel_size=3, stride=1, padding=1)
        # 下采样模块的堆叠
        current_resolution = resolution                                 # 当前的采样块的输入分辨率
        in_feature_dims = (in_dim,) + tuple(feature_dims)[:-1]   # 单个采样块的输入通道数
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            in_dim, out_dim = in_feature_dims[i_level], feature_dims[i_level]   # 采样块的输入输出通道数
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(                                        # 卷积模块
                        in_channels=in_dim,
                        out_channels=out_dim,
                        time_embedding_channels=self.t_embedding_channels,
                        dropout=dropout
                    ))
                in_dim = out_dim
                if current_resolution in attention_resolutions:
                    attn.append(attention_type.make_attn(in_dim))       # 注意力模块

            # 整个下采样模块：卷积->注意力->下采样
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:  # 最后一个下采样模块不需要下采样
                down.downsample = Downsample(in_dim, resamp_with_conv)  # 采样模块
                current_resolution = current_resolution // 2
            self.down.append(down)

        # middle
        # UNet 中间模块：卷积->注意力->卷积
        self.mid = nn.Module()
        self.mid.block_in = ResnetBlock(
            in_channels=in_dim,
            out_channels=in_dim,
            time_embedding_channels=self.t_embedding_channels,
            dropout=dropout)
        self.mid.attention = attention_type.make_attn(in_dim)
        self.mid.block_out = ResnetBlock(
            in_channels=in_dim,
            out_channels=in_dim,
            time_embedding_channels=self.t_embedding_channels,
            dropout=dropout)

        # end
        # 输出层
        self.conv_out = nn.Conv2d(in_dim, 2 * z_channels if any_double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)  # 用于需要获取最后一层的权重
        self.net_out = nn.Sequential(
            Norm(in_dim),
            Swish(),
            self.conv_out
        )

    def forward(self, x):
        # timestep embedding
        time_embedding = None

        # down-sampling
        # ((卷积->注意力)*blocks ->下采样)*layers
        h_skip = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):     # 逐层处理，(卷积->注意力)*n ->下采样
            for i_block in range(self.num_res_blocks):  # 逐块处理，卷积->注意力
                h = self.down[i_level].block[i_block](h_skip[-1], time_embedding)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h_skip.append(h)
            # 记录 skip 连接
            if i_level != self.num_resolutions - 1:     # 下采样，最后一个下采样模块不需要记录到 skip 中
                h_skip.append(self.down[i_level].downsample(h_skip[-1]))

        # middle
        # 卷积->注意力->卷积
        h = h_skip[-1]
        h = self.mid.block_in(h, time_embedding)
        h = self.mid.attention(h)
        h = self.mid.block_out(h, time_embedding)

        # end
        h = self.net_out(h)

        return h

    def get_last_layer(self):
        return self.conv_out.weight


# time-embedding-based Decoder，输入特征图的通道数为 z_channels，输出图像通道数为 out_channels
class Decoder(nn.Module):
    def __init__(self, *,
                 in_channels, z_channels, out_channels, feature_dims=(128, 256, 512, 1024),
                 num_res_blocks, resamp_with_conv=True,
                 resolution, attention_resolutions, attention_type=AttentionType.VANILLA,
                 early_end=False, tanh_out=False,
                 dropout=0.0):
        """
        Args:
            in_channels: 输入通道数
            z_channels: 编码器的输出通道数，也就是解码器的输入通道数，也就是特征图的通道数
            out_channels: 输出通道数
            feature_dims (tuple[int] or list[int]): 各层特征图的通道数
            num_res_blocks: 每层的残差块数
            resamp_with_conv (bool): residual-sampling with convolution 是否使用卷积进行特征上采样
            resolution: 输入图像或特征图的分辨率
            attention_resolutions (tuple[int] or list[int]): 每层注意力模块输入特征图的分辨率
            attention_type: 注意力模块的类型
            early_end: 在最后一层卷积处理之前是否提前输出
            tanh_out: 是否使用 tanh 作为输出的激活函数
            dropout: 卷积层的 dropout 率
        """
        super().__init__()
        self.channels = feature_dims[0]
        self.t_embedding_channels = self.channels * 4
        self.num_resolutions = len(feature_dims)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.early_end = early_end
        self.tanh_out = tanh_out
        self.last_z_shape = None

        # 计算各层的输入通道数
        in_dim = feature_dims[-1]
        current_resolution = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, current_resolution, current_resolution)
        print(f"Working with z of shape {self.z_shape} dimensions.")
        # 通道转换，由输入通道转换为第一个卷积层的输入通道数
        self.conv_in = torch.nn.Conv2d(z_channels, in_dim, kernel_size=3, stride=1, padding=1)

        # middle
        # UNet 中间模块：卷积->注意力->卷积
        self.mid = nn.Module()
        self.mid.block_in = ResnetBlock(
            in_channels=in_dim,
            out_channels=in_dim,
            time_embedding_channels=self.t_embedding_channels,
            dropout=dropout)
        self.mid.attention = attention_type.make_attn(in_dim)
        self.mid.block_out = ResnetBlock(
            in_channels=in_dim,
            out_channels=in_dim,
            time_embedding_channels=self.t_embedding_channels,
            dropout=dropout)

        # up-sampling
        # 上采样模块的堆叠
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            out_dim = feature_dims[i_level]  # 采样块的输入通道数
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(  # 卷积模块
                        in_channels=in_dim,
                        out_channels=out_dim,
                        time_embedding_channels=self.t_embedding_channels,
                        dropout=dropout
                    ))
                in_dim = out_dim
                if current_resolution in attention_resolutions:
                    attn.append(attention_type.make_attn(in_dim))  # 注意力模块

            # 整个上采样模块：卷积->注意力->上采样
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(in_dim, resamp_with_conv)  # 采样模块
                current_resolution = current_resolution * 2
            self.up.insert(0, up)  # 与下采样的顺序相反

        # end
        # 输出层
        self.conv_out = nn.Conv2d(in_dim, out_channels, kernel_size=3, stride=1, padding=1)  # 用于需要获取最后一层的权重
        self.net_out = nn.Sequential(
            Norm(in_dim),
            Swish(),
            self.conv_out
        )

    def forward(self, z, t=None, context=None):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        time_embedding = None

        # z to in_dim
        h = self.conv_in(z)

        # middle
        # 卷积->注意力->卷积
        h = self.mid.block_in(h, time_embedding)
        h = self.mid.attention(h)
        h = self.mid.block_out(h, time_embedding)

        # up-sampling
        # ((卷积->注意力)*blocks ->上采样)*layers
        for i_level in reversed(range(self.num_resolutions)):  # 逐层处理，(卷积->注意力)*n ->上采样
            for i_block in range(self.num_res_blocks + 1):  # 逐块处理，卷积->注意力
                h = self.up[i_level].block[i_block](h, time_embedding)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # 记录 skip 连接
            if i_level != 0:  # 上采样，第一个上采样模块不需要记录到 skip 中
                h = self.up[i_level].upsample(h)

        # early end
        if self.early_end:
            return h

        # end
        h = self.net_out(h)
        if self.tanh_out:
            h = torch.tanh(h)

        return h

    def get_last_layer(self):
        return self.conv_out.weight


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        """
        Args:
            factor: 一个残差卷积模块后特征图的缩放因子
            in_channels: 输入通道数
            mid_channels: 中间通道数
            out_channels: 输出通道数
            depth: 一个残差卷积模块的卷积块数量
        """
        super().__init__()
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        # 残差卷积->注意力->残差卷积
        self.res_block_in = nn.ModuleList([
            ResnetBlock(in_channels=mid_channels, out_channels=mid_channels, time_embedding_channels=0, dropout=0.0)
            for _ in range(depth)])
        self.attn = SpatialSelfAttention(mid_channels)
        self.res_block_out = nn.ModuleList([
            ResnetBlock(in_channels=mid_channels, out_channels=mid_channels, time_embedding_channels=0, dropout=0.0)
            for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.res_block_in:
            x = block(x, None)
        x = F.interpolate(x, size=(int(round(x.shape[2] * self.factor)), int(round(x.shape[3] * self.factor))))
        x = self.attn(x)
        for block in self.res_block_out:
            x = block(x, None)

        x = self.conv_out(x)
        return x

