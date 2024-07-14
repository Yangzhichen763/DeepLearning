import torch
from torch import nn
import math
import torch.nn.functional as F
from modules.attention import MultiheadAttention


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks
    """
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    multi_head_attention = MultiheadAttention(d_model=512, num_heads=8)
    x_input = torch.randn(128, 64, 512)
    x_output, x_attention = multi_head_attention(x_input, x_input, x_input)
    print(x_output.shape)




