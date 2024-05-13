from utils.tensorboard import *

from model import (UNet, UNetCustom)
from model import (UNet_custom_light)
from lately.segment_utils import get_transform, train_and_validate
from utils.pytorch import *


if __name__ == '__main__':
    # 部署 GPU 设备
    device = assert_on_cuda()

    model_creator = UNet_custom_light
    # 训练和验证模型，在 Terminal 激活 tensorboard 的指令:
    # tensorboard --logdir=./legacy/ResNet/logs_tensorboard
    writer = get_writer('./')
    num_classes = train_and_validate(
        transform_image=get_transform(3),
        transform_label=get_transform(1),
        model_creator=model_creator,
        batch_size=8,
        num_samples=[-1, -1],

        learning_rate=0.001,
        scheduler_step_size=2,
        scheduler_gamma=0.5,

        device=device,
        num_epochs=1,
        writer=writer,
    )
    close_writer(writer)

    # 加载最佳模型
    load_model = model_creator(num_classes=10).to(device)
    load.from_model(load_model, f'models/{model_creator.__name__}.pt', device)

    # 测试模型
    # test_model(load_model, x_input, device)
