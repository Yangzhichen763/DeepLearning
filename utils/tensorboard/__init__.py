from torch.utils.tensorboard import SummaryWriter

import os
import warnings
import time

from utils.log.info import print_

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")   # 忽略警告


def get_writer():
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"./logs/tensorboard/{current_time}"
    print_(f"Tensorboard logs will be saved to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir, flush_secs=5)
    return writer


def get_writer_by_name(dir_name, sub_dir_name):
    log_dir = f"./logs/tensorboard/{dir_name}/{sub_dir_name}"
    print_(f"Tensorboard logs will be saved to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir, flush_secs=5)
    return writer


def close_writer(writer):
    writer.close()
