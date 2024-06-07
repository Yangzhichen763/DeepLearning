import torch
from torchvision.transforms import ToPILImage
import numpy as np
import logging

a = torch.rand(5, 3, 16, 16)
b = a.flatten(start_dim=-2)
print(b.shape)

c = [1, 2, 3]
print(tuple(c))
