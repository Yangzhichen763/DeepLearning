import math

import numpy as np
import torch.nn as nn

import torch.nn.init as init

from mlp import AttentionLinear
from modules.attention import SpatialSelfAttention

from testime.mnist import train, test


device = 'cpu'


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class AttentionMLP(nn.Module):
    def __init__(self, input_size: tuple[int, int], output_size, d_model, activations=nn.ReLU()):
        super(AttentionMLP, self).__init__()
        width, height = input_size
        input_dim = width * height

        self.conv = nn.Conv2d(d_model, 1, kernel_size=3, padding=1)
        self.attention = SpatialSelfAttention(d_model)

        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = AttentionLinear(width, width)
        self.fc3 = nn.Linear(width, output_size)

        self.act = activations

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                init.zeros_(m.bias)

    # noinspection PyPep8Naming
    def forward(self, x):
        x = self.conv(x)
        attention, _ = self.attention(x)
        attention = attention.squeeze(dim=1)

        x = x.view(x.shape[0], -1)
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out, attention)
        out = self.act(out)
        out = self.fc3(out)
        return out


if __name__ == '__main__':
    # 超参数设置
    batch_size = 64
    num_workers = 2
    num_epochs = 10
    learning_rate = 3e-4
    input_size = (28, 28)
    hidden_size = 512

    # 训练模型
    _model = AttentionMLP(input_size, output_size=10, d_model=1)
    train(model=_model, num_epochs=num_epochs,
          batch_size=batch_size, learning_rate=learning_rate, device=device, num_workers=num_workers)

    _model = MLP(np.prod(input_size), input_size[0], 10)
    train(model=_model, num_epochs=num_epochs,
          batch_size=batch_size, learning_rate=learning_rate, device=device, num_workers=num_workers)

