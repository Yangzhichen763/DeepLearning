import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

out = torch.randn(1, 8, 4, 4)
spx = torch.split(out, [2, 2, 2, 2], 1)  # 在通道维度上分割张量
print(spx[0].shape, spx[1].shape, spx[2].shape, spx[3].shape)
for i in range(4):  # 对于每个尺度
    if i == 0:
        sp = spx[i]  # 如果是第一个尺度，取出分割后的张量
    else:
        sp = sp + spx[i]  # 否则，累加之前的张量和当前尺度的张量
    sp = torch.zeros(1, 2, 4, 4)
    if i == 0:
        out = sp  # 如果是第一个尺度，直接赋值给out
    else:
        out = torch.cat((out, sp), 1)  # 否则，将当前尺度的结果和之前的结果在通道维度上拼接
out = torch.cat((out, spx[4]), 1)

