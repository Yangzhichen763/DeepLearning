
import torch
from utils.torch import extract


a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(extract(a, 1, a))


