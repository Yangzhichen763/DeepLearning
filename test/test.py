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

a = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
index = torch.tensor([[[0, 1, 2], [1, 0, 1], [2, 2, 0]]])
b = torch.zeros_like(a)
b.scatter_(1, index, a)
print(b)
