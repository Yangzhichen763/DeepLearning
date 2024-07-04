
import torch
import torch.nn.functional as F
from utils.tensorboard import get_writer


a = get_writer()
a.add_scalar('loss', 10, 0)


