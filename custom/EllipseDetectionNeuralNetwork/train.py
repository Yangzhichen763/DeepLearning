import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from custom.EllipseDetectionNeuralNetwork.datasets import EllipseDetectionDataset
from modules.residual import ResNetEncoder, ResNeXtEncoder
from modules.residual.ResNet import BasicBlock, Bottleneck
from optim import CosineAnnealingWarmupRestarts
from utils.general import Trainer, Validator
from utils.tensorboard import *
from utils.pytorch import *
from utils.pytorch.dataset import *
from utils.os import *
from utils import general

import time

from model import *
from loss import *

import cv2


def save_image(pred, target, target_images, batches, ioubs=None):
    """
    Args:
        pred: shape 为 [B, x, 5]
        target: shape 为 [B, x, 5]
        target_images: shape 为 [B, 1, H, W]
        batches:
        ioubs: shape 为 [B, N, x]
    Returns:
    """
    def _draw_ellipse(image, box, color, thickness=2):
        if (box[2:4] < 1).any():
            box[2:4] = box[2:4].clamp(min=1)
        box = box.cpu().numpy()
        box = (box[0:2], box[2:4], box[4] * 180 / math.pi)
        cv2.ellipse(img=image,
                    box=box,
                    color=color,
                    thickness=thickness)

    num_target = target.shape[-2]

    pred = pred.detach()
    target = target.detach()
    target_images = target_images.detach().permute(0, 2, 3, 1).cpu().numpy()

    if ioubs is None:
        ioubs = torch.zeros((batch_size, num_target), dtype=torch.float32, device=device)
    else:
        ioubs, _ = ioubs.detach().max(dim=-1)

    # 如果要绘制原图作为背景，则改为 [(image * 255).astype(np.uint8) for image in target_images]
    images = [np.zeros_like(image) for image in target_images]

    # 绘制椭圆框
    for i, (prrs, trrs, ious) in enumerate(zip(pred, target, ioubs)):   # 每个 batch
        for trr in trrs:                                    # 所有目标框
            _draw_ellipse(images[i], trr,
                          color=(96, 96, 96),
                          thickness=12)
        for j, (prr, iou) in enumerate(zip(prrs, ious)):    # 所有预测框
            _draw_ellipse(images[i], prr,
                          color=(255, 255, 255),
                          thickness=round(iou.item() * 10))

        # writer.add_image(
        #     "train_image", images[i], epoch * batch_size + i, dataformats="HWC"
        # )
        path = get_unique_file_name(
            f"./logs/checkpoints",
            f"pred_{batches * batch_size + i}",
            "jpg",
            unique=False)
        cv2.imwrite(path, images[i])


def train():
    datas = trainer.start(epoch, scheduler=scheduler_epoch)

    _model.train()  # 设置模型为训练模式
    for (i, (images, target)) in datas:
        images, target = images.to(device), target.to(device)

        with torch.cuda.amp.autocast():
            output = _model(images)                          # 前向传播，获得输出
            loss, iou, pred = criterion(output, target)     # 计算损失

        trainer.backward(loss, scheduler=scheduler_batch)

        if epoch % 5 == 0:
            save_image(pred, target, images, i, iou)

        trainer.step(i, loss, scheduler=scheduler_batch,
                     iou=iou.mean(dim=-1).mean(-1).sum())

    trainer.end(scheduler=scheduler_epoch)


def validate():
    datas = validator.start(epoch, optimizer)

    _model.eval()
    with torch.no_grad():
        for (i, (data, target)) in datas:
            data, target = data.to(device), target.to(device)

            output = _model(data)                                # 获得输出
            loss, iou, _ = criterion(output, target)            # 计算损失
            correct = iou.mean(dim=-1).mean(-1).sum()           # 计算正确率

            validator.step(i, loss, correct)

    return validator.end()


def get_model(input_shape, config):
    if config == "ResNet v1.0":
        _in_channels = image_shape[0]
        _dim_classifiers = [64, 16]
        _center_encoder = ResNetEncoder(
            _in_channels,
            BasicBlock,
            num_blocks=[2, 2, 3, 3, 2],
            dim_hidden=[16, 32, 64, 128, 256])
        _size_encoder = ResNetEncoder(
            _in_channels,
            Bottleneck,
            num_blocks=[3, 4, 6, 3],
            dim_hidden=[16, 32, 64, 128])

        _model = EllipseDetectionNetwork_ResNet_v1(
            center_encoder=_center_encoder,
            size_encoder=_size_encoder,
            dim_classifiers=_dim_classifiers,
            device=device
        )
        return _model
    elif config == "ResNeXt v1.0":
        _in_channels = image_shape[0]
        _dim_classifiers = [64, 16]
        _center_encoder = ResNeXtEncoder(
            _in_channels,
            BasicBlock,
            num_blocks=[2, 2, 3, 3, 2],
            dim_hidden=[16, 32, 64, 128, 256])
        _size_encoder = ResNeXtEncoder(
            _in_channels,
            Bottleneck,
            num_blocks=[3, 4, 6, 3],
            dim_hidden=[16, 32, 64, 128])

        _model = EllipseDetectionNetwork_ResNet_v1(
            center_encoder=_center_encoder,
            size_encoder=_size_encoder,
            dim_classifiers=_dim_classifiers,
            device=device
        )
        return _model
    elif config == "ResNeXt v2.0":
        _in_channels = image_shape[0]
        _dim_classifiers = [64, 32]
        _center_encoder = ResNeXtEncoder(
            _in_channels,
            Bottleneck,
            num_blocks=[3, 4, 6, 3],
            dim_hidden=[16, 32, 64, 128])
        _size_encoder = ResNeXtEncoder(
            _in_channels,
            Bottleneck,
            num_blocks=[3, 4, 23, 3],
            dim_hidden=[16, 32, 64, 128])

        _model = EllipseDetectionNetwork_ResNet_v1(
            center_encoder=_center_encoder,
            size_encoder=_size_encoder,
            dim_classifiers=_dim_classifiers,
            device=device
        )
        return _model
    elif config == "AlexNet":
        _model = EllipseDetectionNetwork_AlexNet(
            input_shape=input_shape,
            dim_features=[16, 32, 64, 128, 256],
            num_layers_in_features=[2, 2, 3, 3, 2],
            dim_classifiers=[32, 16],
            device=device
        )
        return _model


def get_scheduler(config):
    if config == "StepLR":
        _scheduler_epoch = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma)
        _scheduler_batch = None
    elif config == "CosineAnnealingWarmRestarts":
        _scheduler_epoch = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(num_epochs / 40),
            max_lr=learning_rate,
            min_lr=1e-8,
            warmup_steps=5,
            gamma=0.71)
        _scheduler_batch = None
    elif config == "CyclicLR":
        _scheduler_epoch = None
        _scheduler_batch = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=0.0005, max_lr=0.01,
                step_size_up=100, mode='triangular2',
                cycle_momentum=False)
    else:
        _scheduler_epoch = None
        _scheduler_batch = None

    return _scheduler_epoch, _scheduler_batch


if __name__ == '__main__':
    # 超参数设置
    batch_size = 32
    num_epochs = 1000
    num_workers = 4  # os.cpu_count()
    learning_rate = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    scheduler_step_size = 5
    scheduler_gamma = 0.5

    # batch_size = 8
    # _position = torch.randn(batch_size, 1, 2) * 32
    # _position_offset = torch.randn(batch_size, 1, 2) * 32
    # _size = torch.randn(batch_size, 1, 2) * 16 - 24
    # _offset_size = torch.randn(batch_size, 1, 2) * 64
    # for k in range(60):
    #     print(k)
    #     _target_image = torch.zeros((batch_size, 1, 256, 256))
    #
    #     _position_pred = _position + _position_offset
    #     _size_pred = _size + _offset_size
    #     _angle_pred = torch.tensor([float(6 * k * math.pi / 180)]).repeat(batch_size, 1, 1)
    #     _pred = torch.cat([_position_pred, _size, _angle_pred], dim=-1)
    #
    #     _angle_target = torch.tensor([0]).repeat(batch_size, 1, 1)
    #     _target = torch.cat([_position, _size, _angle_target], dim=-1)
    #     _target = EllipseLoss.regress_rr(_target, _target_image.shape)
    #
    #     _loss, _iou, _pred = EllipseLoss((256, 256))(_pred, _target)
    #     save_image(_pred, _target, _target_image, k, _iou)
    # exit()

    # 部署 GPU 设备
    device = assert_on_cuda()

    writer = get_writer()

    # 加载数据集
    print(f"Using {num_workers} workers")
    dataset = EllipseDetectionDataset(data_set="1x")
    train_set, val_set = datapicker.split_train_test(dataset, [0.8, 0.2], random=False)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(
        val_set,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True, pin_memory=True)

    # 定义模型
    image_shape = (1, 256, 256)
    model_config = "ResNeXt v1.0"
    model = get_model(image_shape, model_config)
    model.to(device)

    # 定义损失函数和优化器
    criterion = EllipseLoss(image_shape)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate)
    scheduler_config = "CosineAnnealingWarmRestarts"
    scheduler_epoch, scheduler_batch = get_scheduler(scheduler_config)

    # 训练模型
    # 训练和验证模型，在 Terminal 激活 tensorboard 的指令:
    # tensorboard --logdir=./custom/EllipseDetectionNeuralNetwork/logs/tensorboard
    trainer = Trainer(train_loader, optimizer, writer=writer)
    validator = Validator(test_loader)
    best_accuracy, last_accuracy = 0.0, 0.0
    for epoch in range(1, num_epochs + 1):
        train()
        _, last_accuracy = validate()
        if last_accuracy > best_accuracy:
            best_accuracy = last_accuracy
            save.as_pt(model, "models/best.pt")
        save.as_pt(model, "models/last.pt")

    # 打印训练结果
    final_learning_rate = optimizer.param_groups[0]['lr']
    print(f"\nSummary: "
          f"\n - [Model Config]: {model_config}  [scheduler_config]: {scheduler_config} "
          f"\n - [Total Epochs]: {num_epochs} "
          f"\n - [Batch Size]: {batch_size} "
          f"\n - [Final Learning Rate]: {final_learning_rate} "
          f"\n - [Best Accuracy]: {best_accuracy:.2f}%  [Last Accuracy]: {last_accuracy:.2f}%")

    close_writer(writer)
