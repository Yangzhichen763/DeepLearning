import torch
from torch import nn
import torch.nn.functional as F

from utils.logger.modellogger import *


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product attention
    """
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: query （要查询的信息）
        :param k: key   （被查询的向量）
        :param v: value （查询得到的值）
        :param mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        :return:
        """
        # q @ k.transpose(2, 3) 得到的矩阵可以用来表示 attention 强度
        # / self.temperature 是为了防止内积过大导致偏导数趋近于 0（可以让注意力的分布更加均匀）
        attention = (q / self.temperature) @ k.transpose(2, 3)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        # dim=-1 表示在最后一个维度上应用 softmax 函数，前面几个维度保持不变
        attention = self.dropout(F.softmax(attention, dim=-1))
        output = attention @ v
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
        :param x: 输入向量
        :param mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        :return:
        """
        batch_size, time, dimension = x.shape
        x = x.view(batch_size, time, 1, dimension)
        x, attention = self.attention(x, x, x, mask=mask)
        x = x.view(batch_size, time, dimension)
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
        :param q: query （要查询的信息）
        :param k: key   （被查询的向量）
        :param v: value （查询得到的值）
        :param mask: 可以是右上角遮罩（右上角都是 0） mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        :return:
        """
        batch_size, time, dimension = q.shape
        n_dim = self.d_model // self.num_heads

        # 获得 Q, K, V 矩阵
        q = self.weight_Q(q).view(batch_size, time, self.num_heads, n_dim).permute(0, 2, 1, 3)
        k = self.weight_K(k).view(batch_size, time, self.num_heads, n_dim).permute(0, 2, 1, 3)
        v = self.weight_V(v).view(batch_size, time, self.num_heads, n_dim).permute(0, 2, 1, 3)

        # 计算注意力权重
        q, attention = self.attention(q, k, v, mask=mask)

        # 合并输出
        q = q.permute(0, 2, 1, 3).contiguous().view(batch_size, time, dimension)
        q = self.dropout(self.weight_combine(q))
        return q, attention

