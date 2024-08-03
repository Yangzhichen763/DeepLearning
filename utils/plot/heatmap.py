import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


# 生成 ERF 图，参考：https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/erf/analyze_erf.py

def show_heatmaps(
        matrices,
        x_label,
        y_label,
        titles=None,
        figure_size=(2.5, 2.5),
        cmap='Reds'):
    pass