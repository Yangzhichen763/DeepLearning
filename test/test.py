import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import numpy as np
import logging

a: torch.Tensor = torch.rand(3 * 5, 32, 16)
c = a.unflatten(dim=0, sizes=(-1, 5))
print(c.shape)


