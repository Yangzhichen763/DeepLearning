from utils.tensorboard import *
import torchvision.transforms as transforms

from model import (VGG11, VGG13, VGG16, VGG19)
from legacy.classify_utils import train_and_validate
from utils.pytorch import *


def get_transform():
    """
    获取数据预处理器
    :return:
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 从 (bath_size, 3, 32, 32) 缩放到 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform


if __name__ == '__main__':
    # 部署 GPU 设备
    device = assert_on_cuda()

    model_creator = VGG11
    # 训练和验证模型，在 Terminal 激活 tensorboard 的指令:
    # tensorboard --logdir=./lately/UNet/logs_tensorboard
    writer = get_writer()
    num_classes = train_and_validate(
        get_transform(),
        model_creator=model_creator,
        batch_size=16,
        num_samples=[16000, 4000],

        learning_rate=0.001,
        scheduler_step_size=2,
        scheduler_gamma=0.5,

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
