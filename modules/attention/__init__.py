import torch
import torchvision.datasets.voc
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from utils.log.model import *


"""
Attention is All You Need
论文链接 2017-2023：https://arxiv.org/abs/1706.03762
"""


class AdditiveAttention(nn.Module):
    """
    Additive attention
    一般来说，当 query 和 key 维度不同时，可以使用 additive attention。
    论文链接 2014-2016：https://arxiv.org/abs/1409.0473
    """
    def __init__(self, dim_hidden, q_dim, k_dim, dropout=0.1):
        super(AdditiveAttention, self).__init__()
        self.q_weight = nn.Linear(q_dim, dim_hidden, bias=False)
        self.k_weight = nn.Linear(k_dim, dim_hidden, bias=False)
        self.v_weight = nn.Linear(dim_hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: query （要查询的信息）[batch_size, q_n, q_dim]
        :param k: key   （被查询的向量）[batch_size, kv_n, k_dim]
        :param v: value （查询得到的值）[batch_size, kv_n, v_dim]
        :param mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        :return:
        """
        q = self.q_weight(q)     # [batch_size, q_n, q_dim] -> [batch_size, q_n, dim_hidden]
        k = self.k_weight(k)     # [batch_size, kv_n, k_dim] -> [batch_size, kv_n, dim_hidden]

        # 加性融合
        # q: [batch_size, q_n, dim_hidden] -> [batch_size, q_n, 1, dim_hidden]
        # k: [batch_size, kv_n, dim_hidden] -> [batch_size, 1, kv_n, dim_hidden]
        # energy: [batch_size, q_n, kv_n, dim_hidden]
        energy = torch.tanh(q.unsqueeze(-2) + k.unsqueeze(-3))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        # v_weight 只有一个输出，所以要移除左后一个维度
        scores = self.v_weight(energy).squeeze(-1)  # [batch_size, q_n, kv_n, 1] -> [batch_size, q_n, kv_n]
        attention = F.softmax(scores, dim=-1)       # [batch_size, q_n, kv_n]
        attention = self.dropout(attention)

        # [batch_size, q_n, kv_n] @ [batch_size, kv_n, v_dim] -> [batch_size, q_n, v_dim]
        output = torch.bmm(attention, v)
        return output, attention


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product attention
    使用点积可以得到计算效率更高的评分函数，但是点积操作要求 query 和 key 具有相同的维度 d
    """
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: query （要查询的信息）[batch_size, q_n, dim]
        :param k: key   （被查询的向量）[batch_size, kv_n, dim]
        :param v: value （查询得到的值）[batch_size, kv_n, dim]
        :param mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(q_n, kv_n, dtype=torch.bool))
        :return:
        """
        # q @ k.transpose(-2, -1) 得到的矩阵可以用来表示 attention 强度
        # / self.temperature 是为了防止内积过大导致偏导数趋近于 0（可以让注意力的分布更加均匀）
        # [batch_size, q_n, dim] @ [batch_size, dim, kv_n] -> [batch_size, q_n, kv_n]
        attention = torch.bmm((q / self.temperature), k.transpose(-2, -1))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        # dim=-1 表示在最后一个维度上应用 softmax 函数，前面几个维度保持不变
        attention = self.dropout(F.softmax(attention, dim=-1))
        # torch.bmm 在处理批量矩阵乘法的性能比 torch.matmul（或者 @ 运算符）要好
        # [batch_size, q_n, kv_n] @ [batch_size, kv_n, dim] -> [batch_size, q_n, dim]
        output = torch.bmm(attention, v)
        return output, attention


class SelfAttention(nn.Module):
    """
    Self-attention
    """
    def __init__(self, d_model, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5, dropout=dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入向量
            mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        Returns:
        """
        batch_size, time, dimension = x.shape
        x = x.view(batch_size, time, 1, dimension)
        x, attention = self.attention(x, x, x, mask=mask)
        x = x.view(batch_size, time, dimension)
        return x, attention


class CrossAttention(nn.Module):
    """
    Cross Attention
    与 Scale Dot-Product Attention 、Self Attention 的区别：
    1. Cross Attention -> query, (key, value)
    2. Self Attention -> (query, key, value)
    3. Scaled Dot-Product Attention -> query, key, value
    """
    def __init__(self, d_model, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5, dropout=dropout)

    def forward(self, query, context, mask=None):
        """
        Args:
            query: 输入向量
            context: 上下文向量
            mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        Returns:
        """
        x, attention = self.attention(query, context, context, mask=mask)
        return x, attention


class MultiheadAttention(nn.Module):
    """
    Multi-Head attention
    """
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        self.weight_Q = nn.Linear(d_model, d_model, bias=False)
        self.weight_K = nn.Linear(d_model, d_model, bias=False)
        self.weight_V = nn.Linear(d_model, d_model, bias=False)
        self.weight_combine = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: query （要查询的信息）[batch_size, q_n, dim]
        :param k: key   （被查询的向量）[batch_size, kv_n, dim]
        :param v: value （查询得到的值）[batch_size, kv_n, dim]
        :param mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(q_n, kv_n, dtype=torch.bool))
        :return:
        """
        batch_size, _, dimension = q.shape
        n_dim = self.d_model // self.num_heads

        # 获得 Q, K, V 矩阵
        q = self.weight_Q(q).view(batch_size, -1, self.num_heads, n_dim).permute(0, 2, 1, 3)
        k = self.weight_K(k).view(batch_size, -1, self.num_heads, n_dim).permute(0, 2, 1, 3)
        v = self.weight_V(v).view(batch_size, -1, self.num_heads, n_dim).permute(0, 2, 1, 3)

        # 计算注意力权重
        q, attention = self.attention(q, k, v, mask=mask)

        # 合并输出
        q = q.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, dimension)
        q = self.dropout(self.weight_combine(q))
        return q, attention


class VisionAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight_Q = nn.Conv2d(d_model, d_model, 1, stride=1, padding=0)
        self.weight_K = nn.Conv2d(d_model, d_model, 1, stride=1, padding=0)
        self.weight_V = nn.Conv2d(d_model, d_model, 1, stride=1, padding=0)
        self.weight_Output = nn.Conv2d(d_model, d_model, 1, stride=1, padding=0)
        self.initialize()

        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5, dropout=0)

    def initialize(self):
        for module in [self.weight_Q, self.weight_K, self.weight_V, self.weight_Output]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.weight_Output.weight, gain=1e-5)

    # noinspection PyPep8Naming
    def forward(self, x):
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        # 先 permute 或者 transpose 再 view 需要注意是否会影响 view ，否则要在中间添加 contiguous
        q = self.weight_Q(x).permute(0, 2, 3, 1).view(B, H * W, C)
        k = self.weight_K(x).permute(0, 2, 3, 1).view(B, H * W, C)
        v = self.weight_V(x).permute(0, 2, 3, 1).view(B, H * W, C)

        # 计算注意力权重
        output, attention = self.attention(q, k, v)

        # [B, HxW, C] -> [B, H, W, C] -> [B, C, H, W]
        output = output.view(B, H, W, C).permute(0, 3, 1, 2)
        output = self.weight_Output(output)

        return x + output, attention


if __name__ == '__main__':
    _model = CrossAttention(d_model=512)
    _query = torch.rand(10, 20, 512)
    _context = torch.rand(10, 30, 512)
    _output, _attention = _model(_query, _context)
    print(_output.shape)
    print(_attention.shape)

