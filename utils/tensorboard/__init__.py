from torch.utils.tensorboard import SummaryWriter

import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")   # 忽略警告


def get_writer(log_dir):
    log_dir = os.path.join(log_dir, 'logs/tensorboard')
    print(f"Tensorboard logs will be saved to {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def close_writer(writer):
    writer.close()