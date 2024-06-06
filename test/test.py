import torch
from torchvision.transforms import ToPILImage
import numpy as np
import logging

# 参数中 level 代表 INFO 即以上级别的日志信息才能被输出
logging.basicConfig(format="\n%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

image_file_name = "image.png"
b = image_file_name[:image_file_name.rindex('.')] + "a" + image_file_name[image_file_name.rindex('.'):]
logging.info(b)
