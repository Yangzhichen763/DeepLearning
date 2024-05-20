import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
from PIL import Image

import math

from custom.EllipseDetectionNeuralNetwork.loss import EllipseLoss

loss_func = nn.CrossEntropyLoss()
pre = torch.tensor([0.8, 0.5, 0.2, 0.5], dtype=torch.float)
tgt = torch.tensor([1, 0, 0, 0], dtype=torch.float)
loss = loss_func(pre, tgt)
print(loss)
