import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
from PIL import Image

a = torch.randn(3, 256, 256).cpu().permute(1, 2, 0)
b = a.numpy()
print(b.shape)
image = Image.fromarray(b, mode='RGB')
if not os.path.exists("./output"):
    os.makedirs("./output")
image.save(f"./output/1.png")
print(b.shape)
