import enum
import math

import torch
import torch.nn as nn


class ParameterInitType(enum.Enum):
    ZERO = enum.auto()
    XAVIER_UNIFORM = enum.auto()
    XAVIER_NORMAL = enum.auto()


# 参数初始化，对参数 parameter，使用 init_type 种类的初始化方法进行初始化
def init_parameters(parameter, init_type):
    """
    Initialize the parameters of a module using the specified initialization type.

    Args:
        parameter (nn.Parameter): The parameter to be initialized.
        init_type (ParameterInitType): The initialization type to use.
    """
    if init_type == ParameterInitType.ZERO:
        nn.init.zeros_(parameter)
    elif init_type == ParameterInitType.XAVIER_UNIFORM:
        nn.init.xavier_uniform_(parameter)
    elif init_type == ParameterInitType.XAVIER_NORMAL:
        nn.init.xavier_normal_(parameter)
    else:
        raise NotImplementedError(f"Unsupported parameter initialization type: {init_type}")


# == Embedding Modules ==


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embedding = None


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


# 随机位置编码，和 Learnable Positional Embedding 结合形成单纯的可学习位置编码
#   - 简单方便，无需额外开销
#   - 位置编码是独立训练到的，不同位置的编码向量没有明显的约束关系，因此只能建模**绝对位置信息**，不能建模相对位置信息
#   - 输入长度不能超过位置编码范围
class RandomPositionalEmbedding(Embedding):
    """
    随机位置编码，最好和 Learnable Positional Embedding 结合使用
    比如：LearnablePositionalEncoding(RandomPositionalEmbedding(d_model, max_length), FFN)
    """
    def __init__(self,
                 d_model: int,
                 max_length: int,
                 init_type: ParameterInitType = ParameterInitType.XAVIER_UNIFORM,
                 device=None):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        position_embedding = torch.empty((max_length, d_model), device=device)
        self.embedding = nn.Parameter(position_embedding, requires_grad=False)

        # Initialize the embedding parameters
        init_parameters(self.embedding, init_type)

    def forward(self, x):
        seq_len = x.shape[1]
        return self.embedding[:seq_len, :]


# Sinusoidal 相对位置编码，可区别位置关系但无法区别前后关系，是 Transformer 模型中用到的位置编码模式
#   - Sinusoidal Positional Encoding 和结合了 Random Positional Embedding 的 Learnable Positional Embedding 在实验表现上区别不大
class SinusoidalPositionalEmbedding1d(Embedding):
    """
    一维相对位置编码，可区别位置关系但无法区别前后关系
    代码修改于：https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    """
    def __init__(self, d_model: int, max_length: int, device=None):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even (got d_model dim = {d_model})"

        position_embedding = torch.zeros((max_length, d_model), device=device)

        position = torch.arange(0, max_length, dtype=torch.float, device=device).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float, device=device)
        theta = torch.exp(_2i * -(math.log(10000.0) / d_model))

        # Even dimensions use sine and odd dimensions use cosine
        position_embedding[:, 0::2] = torch.sin(position * theta)
        position_embedding[:, 1::2] = torch.cos(position * theta)

        self.embedding = nn.Parameter(position_embedding, requires_grad=False)

    def forward(self, x):
        length = x.shape[-2]
        return self.embedding[:length, :]


class SinusoidalPositionalEmbedding2d(Embedding):
    """
    二维相对位置编码，可区别位置关系但无法区别前后关系
    代码修改于：https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    """
    def __init__(self, d_model: int, height: int, width: int, device=None):
        super().__init__()
        assert d_model % 4 == 0, f"d_model must be divisible by 4 (got d_model dim = {d_model})"

        position_embedding = torch.zeros((d_model, height, width), device=device)
        d_model = d_model // 2     # 使用 d_model 的一半作为向量的每个维度

        _2i = torch.arange(0, d_model, step=2, dtype=torch.float, device=device)
        theta = torch.exp(_2i * -(math.log(10000.0) / d_model))
        position_width = torch.arange(0, width, dtype=torch.float, device=device).unsqueeze(1)
        position_height = torch.arange(0, height, dtype=torch.float, device=device).unsqueeze(1)

        position_embedding[0:d_model:2, :, :] = (torch.sin(position_width * theta)
                                             .transpose(0, 1)
                                             .unsqueeze(1)
                                             .repeat(1, height, 1))
        position_embedding[1:d_model:2, :, :] = (torch.cos(position_width * theta)
                                             .transpose(0, 1)
                                             .unsqueeze(1)
                                             .repeat(1, height, 1))
        position_embedding[d_model::2, :, :] = (torch.sin(position_height * theta)
                                            .transpose(0, 1)
                                            .unsqueeze(2)
                                            .repeat(1, 1, width))
        position_embedding[d_model + 1::2, :, :] = (torch.cos(position_height * theta)
                                                .transpose(0, 1)
                                                .unsqueeze(2)
                                                .repeat(1, 1, width))

        self.embedding = nn.Parameter(position_embedding, requires_grad=False)

    def forward(self, x):
        return self.embedding


class RotaryPositionalEmbedding(Embedding):
    def __init__(self, d_model: int, max_length: int, device=None):
        super().__init__()
        self.spe = SinusoidalPositionalEmbedding1d(d_model, max_length, device)
        self.embedding = self.spe.embedding

    def forward(self, x):
        embedding = self.spe(x)
        repeat_shape = (*x.shape[:-len(embedding.shape)], *([1] * len(embedding.shape)))  # (B, N, C) -> (B, 1, 1)
        embedding = embedding.repeat(repeat_shape)                                        # [N, C] -> [B, N, C]
        cos_position = embedding[..., 1::2].repeat_interleave(2, dim=-1)
        sin_position = embedding[..., 0::2].repeat_interleave(2, dim=-1)

        # stack + reshape != cat, 不能使用 cat 代替，但是 stack(dim=n) + reshape == cat(dim=n+1)
        # stack(dim=n) + reshape 可以达到交替的效果(其中 reshape 的 shape 和 cat(dim=n) 的 shape 度相同）
        # 比如：stack([[1, 2], [-1, -2]], dim=-1).reshape -> [1, -1, 2, -2]
        _x = (torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1)
              .reshape(x.shape))
        x = x * cos_position + _x * sin_position    # 相当于将 x 进行旋转

        return x


class RelativePositionalEmbedding(Embedding):
    """
    Relative Position Embeddings Module
    代码修改于：https://github.com/AliHaiderAhmad001/Self-Attention-with-Relative-Position-Representations/blob/main/relation_aware_attention.py

    This module generates learnable relative position embeddings to enrich
    the self-attention mechanism with information about the relative distances
    between elements in input sequences.

    Args:
        d_model (int): Number of dimensions in the relative position embeddings.
        clipping_distance (int): Clipping distance.

    Attributes:
        embedding (nn.Parameter): Learnable parameter for relative position embeddings.

    Example:
        >>> # Create a RelativePosition instance with 16 dimensions and clipping distance of 10
        >>> relative_position = RelativePositionalEmbedding(d_model=16, clipping_distance=10)
        >>> # Generate relative position embeddings for sequences of lengths 5 and 7
        >>> embeddings = relative_position(length_query=5, length_key=7)
    """

    def __init__(self,
                 d_model: int,
                 clipping_distance: int,
                 init_type: ParameterInitType = ParameterInitType.XAVIER_UNIFORM,
                 device=None):
        """
        Initialize the RelativePosition module.

        Args:
            d_model (int): Number of dimensions in the relative position embeddings.
            clipping_distance (int): Clipping distance.
        """
        super().__init__()
        self.d_model = d_model
        self.clipping_distance = clipping_distance

        embedding = torch.empty((2 * clipping_distance + 1, d_model), device=device)
        self.embedding = nn.Parameter(embedding)

        # Initialize the embedding parameters
        init_parameters(self.embedding, init_type)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position embeddings.

        Args:
            q (torch.Tensor): query.
            kv (torch.Tensor): key or value.

        Returns:
            embeddings (torch.Tensor): Relative position embeddings (length_query, length_key, embedding_dim).
        """
        # Generate relative position embeddings
        indices_q = torch.arange(q.shape[1], device=self.embedding.device)
        indices_k = torch.arange(kv.shape[1], device=self.embedding.device)

        distance_matrix = indices_k[None, :] - indices_q[:, None]
        distance_matrix_clipped = torch.clamp(distance_matrix, -self.clipping_distance, self.clipping_distance)
        final_matrix = distance_matrix_clipped + self.clipping_distance

        embeddings = self.embedding[final_matrix.to(torch.long)]
        return embeddings


class LearnablePositionalEmbedding(Embedding):
    def __init__(self, encoding, ffn):
        super().__init__()
        self.encoding: nn.Module = encoding
        self.ffn = ffn

        # 将名为 embedding 的参数的 requires_grad 设置为 True
        for name, parameter in self.encoding.named_parameters():
            if name == "embedding":
                parameter.requires_grad = True

    def forward(self, x):
        x = self.encoding(x)
        x = self.ffn(x)
        return x


class RotaryEmbedding(Embedding):
    """
    代码来源于 https://github.com/test-time-training/ttt-lm-pytorch
    """
    def __init__(
        self,
        dim,
        max_position_embeddings=16,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


if __name__ == '__main__':
    a = torch.randn(2, 3, 4, 5)
    _sum = torch.sum(a, dim=0)
    print(_sum)
