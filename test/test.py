import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
from PIL import Image

import math

from custom.EllipseDetectionNeuralNetwork.loss import EllipseLoss
from modules.residual.ResNet import resnet18
from optim import CosineAnnealingWarmupRestarts


def get_boo():
    list = [4, 5, 6]
    return 1, 2, *list

print(get_boo())

