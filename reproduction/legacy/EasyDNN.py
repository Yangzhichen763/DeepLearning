import torch
import torch.nn as nn
import onnx
import numpy as np


class EasyDNN(nn.Module):
    def __init__(self):
        """ 搭建神经网络 Graph """
        super(EasyDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 5), nn.ReLU(),
            nn.Linear(5, 5), nn.ReLU(),
            nn.Linear(5, 5), nn.ReLU(),
            nn.Linear(5, 3)
        )

    def forward(self, x):
        """ 前向传播 """
        y = self.net(x)
        return y


class EasyMLP(nn.Module):
    """
    简单的单层多层感知机（MLP）
    """
    def __init__(self):
        super(EasyMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        y = self.net(x)
        return y


model = EasyDNN().to('cuda:0')
model.eval()

input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, 'model.onnx', input_names=input_names, output_names=output_names)
