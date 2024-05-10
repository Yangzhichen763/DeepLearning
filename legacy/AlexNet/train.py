
from torch.utils.tensorboard import SummaryWriter

from model import AlexNet_CIFAR10
from legacy.classify_utils import get_transform, train_and_validate
from utils.pytorch import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # 部署 GPU 设备
    device = assert_on_cuda()

    model_creator = AlexNet_CIFAR10
    # 训练和验证模型，在 Terminal 激活 tensorboard 的指令:
    # tensorboard --logdir=./legacy/AlexNet/logs_tensorboard
    writer = SummaryWriter(log_dir='./logs_tensorboard')
    num_classes = train_and_validate(
        get_transform(),
        model_creator=model_creator,
        batch_size=16,
        num_samples=-1,

        learning_rate=0.001,
        scheduler_step_size=1,
        scheduler_gamma=0.75,

        device=device,
        num_epochs=10,
        writer=writer,
    )
    writer.close()

    # 加载最佳模型
    load_model = model_creator(num_classes=num_classes).to(device)
    load.from_model(load_model, f'models/{model_creator.__name__}.pt', device)

    # 测试模型
    # test_model(load_model, x_input, device)
