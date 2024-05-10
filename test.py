import numpy as np
import torch
import torch.nn as nn
import os

path = './datas/documents'
full_path = os.path.expanduser(path)
print(full_path)
if os.path.exists(full_path):
    print("The path exists")
else:
    os.makedirs(full_path)
    print("The path does not exist")







