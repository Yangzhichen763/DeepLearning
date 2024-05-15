import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
from PIL import Image

a = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 4]]).squeeze()
b, c, d = a[[0, 1, 2, 2]], a[[0, 2, 1, 1]], a[[1, 2, 0, 1]]
e = torch.stack([b, c, d])
f = e[[0, 2, 1]].permute(1, 2, 0)
print(f.shape)
h = torch.flatten(f, start_dim=0, end_dim=1)
h = torch.unique(h, dim=0)
print(h)
