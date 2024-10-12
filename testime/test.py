
import torch
import math


n = 3
h = w = 224
p = 16
c = 28
x = torch.randn(n, h*w//(p*p), p*p*c)
print(x.shape)
x = torch.einsum()
print(x.shape)


