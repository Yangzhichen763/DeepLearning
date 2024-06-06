import torch
from torchvision.transforms import ToPILImage
import numpy as np

image_file_name = "image.png"
b = image_file_name[:image_file_name.rindex('.')] + "a" + image_file_name[image_file_name.rindex('.'):]
print(b)
