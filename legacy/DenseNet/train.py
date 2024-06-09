
from utils.tensorboard import *

from model import DenseNet121
from legacy.classify_utils import get_transform, train_and_validate
from utils.torch import *


if __name__ == '__main__':
    # 部署 GPU 设备
    device = assert_on_cuda()

    model_creator = DenseNet121
    # 训练和验证模型，在 Terminal 激活 tensorboard 的指令:
    # tensorboard --logdir=./legacy/AlexNet/logs_tensorboard
    writer = get_writer()
    num_classes = train_and_validate(
        get_transform(),
        model_creator=model_creator,
        batch_size=16,
        num_samples=[-1, -1],

        optimizer_type='SGD',
        learning_rate=0.001,
        weight_decay=1e-4,
        momentum=0.9,
        scheduler_step_size=1,
        scheduler_gamma=0.75,

        device=device,
        num_epochs=10,
        writer=writer,
    )
    close_writer(writer)

    # 加载最佳模型
    load_model = model_creator(num_classes=num_classes).to(device)
    load.from_model(load_model, device, f'models/{model_creator.__name__}.pt')

    # 测试模型
    # test_model(load_model, x_input, device)