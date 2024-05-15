from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

        scheduler_step_size=3,
        scheduler_gamma=0.75,

        device=device,
        num_epochs=30,
        writer=writer,
    )
    close_writer(writer)

    # 加载最佳模型
    load_model = model_creator(num_classes=21).to(device)
    load.from_model(load_model, f'models/{model_creator.__name__}.pt', device)

    # 测试模型
    # test_model(load_model, x_input, device)


def carvana():
    image_size = (256, 256)
    train_transform = A.Compose([
        A.Resize(*image_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(*image_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    # train_loader, val_loader = get_carvana_loaders(
