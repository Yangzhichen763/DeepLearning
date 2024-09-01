
import torch
import math


a = torch.randn(1, 3, 6)
b = torch.randn(1, 3, 4)
c = torch.cat([a, b], dim=2)
print(c.shape)


