import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.init as init
from tqdm import tqdm

from utils.os import get_root_path
from utils.general import Trainer, Manager, Validator

from mlp import AttentionLinear
from modules.attention import SpatialSelfAttention


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


# noinspection PyUnresolvedReferences
def train(model, num_epochs, model_name='mlp_variant', ckpt_path=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # 加载训练集
    train_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 加载测试集
    test_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    trainer = Trainer(train_loader, optimizer)
    manager = Manager(model, device)
    validator = Validator(test_loader)
    for epoch in range(1, num_epochs + 1):
        train_datas = trainer.start(epoch, model=model)
        for (i, (x, y)) in train_datas:
            x = x.to(device)

            predict, loss = trainer.predict(model, x, y, criterion)  # 预测和计算损失
            trainer.backward(loss)  # 反向传播
            trainer.step(i, loss)   # 更新参数
        trainer.end()

        test_datas = validator.start(epoch, model=model, optimizer=optimizer)
        with torch.no_grad():
            for (i, (x, y)) in test_datas:
                x = x.to(device)

                logit, loss = validator.predict(model, x, y, criterion)  # 预测和计算损失

                _, predict = torch.max(logit, 1)
                num_correct = (predict == y).sum().item()
                validator.step(i, loss, num_correct)    # 记录准确率和损失
        average_loss, accuracy = validator.end()

        manager.update_checkpoint(accuracy)


# noinspection PyUnresolvedReferences
def test(model, x):
    tester = Tester(x, model=model)
    with torch.no_grad():
        x = x.to(device)

        logit, loss = tester.predict(model, x, y, criterion)  # 预测和计算损失

        # 计算准确率
        _, predict = torch.max(logit, 1)
        return predict[0]


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
    train(_model, num_epochs)

    _model = MLP(np.prod(input_size), input_size[0], 10)
    train(_model, num_epochs)

