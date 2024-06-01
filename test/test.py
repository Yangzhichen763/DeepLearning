import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
from PIL import Image

import math

from custom.EllipseDetectionNeuralNetwork.loss import EllipseLoss

a = torch.tensor([[1, 2, 3, 4], [4, 2, 3, 1]])
b = torch.tensor([[[1, 1.2], [3, 3.2], [4, 5.3], [2.1, 2.3]], [[1.2, 5], [1.3, 2], [4, 9.2], [2, 3]]])

a_argmin = torch.tensor([0, 3])

result = torch.gather(b, 1, a_argmin.view(-1, 1, 1).expand(-1, 1, 2))

print(result)
