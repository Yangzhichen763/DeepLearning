from torch.utils.tensorboard import SummaryWriter

import os
import warnings
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")   # 忽略警告


def get_writer(filename_suffix=""):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"./logs/tensorboard/{current_time}"
    print(f"Tensorboard logs will be saved to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix, flush_secs=5)
    return writer


def close_writer(writer):
    writer.close()