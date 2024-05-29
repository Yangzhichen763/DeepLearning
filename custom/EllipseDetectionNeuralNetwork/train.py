import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from custom.EllipseDetectionNeuralNetwork.datasets import EllipseDetectionDataset
from modules.residual import ResNetEncoder, ResNeXtEncoder
from modules.residual.ResNet import BasicBlock, Bottleneck
from utils.tensorboard import *
from utils.pytorch import *
from utils.pytorch.dataset import *
from utils.os import *

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

        path = get_unique_file_name(
            f"./logs/checkpoints",
            f"pred_{batches * batch_size + i}",
            "jpg",
            unique=False)
        cv2.imwrite(path, images[i])


def train():
    min_loss = float('inf')
    max_loss = float('-inf')
    total_loss = 0
    dataset_size = len(train_loader.dataset)
    dataset_batches = len(train_loader)

    tqdm.write(f"\nEpoch {epoch}: [learning rate: {scheduler.get_last_lr()[0]}]")
    datas = buffer_dataloader(train_loader)

    model.train()  # 设置模型为训练模式
    with tqdm(
            total=dataset_size,
            desc=f"Training Epoch {epoch}",
            unit='image') as pbar:
        for (i, (images, target)) in datas:
            images, target = images.to(device), target.to(device)

            with torch.cuda.amp.autocast():
                output = model(images)                          # 前向传播，获得输出
                loss, iou, pred = criterion(output, target)     # 计算损失

            optimizer.zero_grad()                           # 清空梯度
            scaler.scale(loss).backward()                   # 反向传播
            scaler.step(optimizer)                          # 更新参数
            scaler.update()                                 # 更新 GradScaler

            if epoch % 5 == 0:
                save_image(pred, target, images, i, iou)

            # 记录损失最小值和最大值以及总损失
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())
            total_loss += loss.item()

            # 打印训练进度
            i_current_batch = i + 1
            pbar.set_postfix(loss=loss.item())
            pbar.update(batch_size)

            if writer is not None:
                writer.add_scalar('./logs/tensorboard/loss', loss.item(), i_current_batch)

        pbar.set_postfix()                  # 清空进度条备注
        pbar.update(pbar.total - pbar.n)    # 防止进度条超过 100%

        tqdm.write(
            f"Min-Avg-Max loss: [{min_loss:.4f}, {total_loss / len(train_loader):.4f}, {max_loss:.4f}]")


def validate():
    total_loss = 0
    correct = 0.
    dataset_size = len(test_loader.dataset)
    dataset_batches = len(test_loader)

    datas = buffer_dataloader(test_loader)

    model.eval()
    with tqdm(
            total=dataset_size,
            desc=f"Testing Epoch {epoch}",
            unit='image') as pbar, torch.no_grad():
        for (_, (data, target)) in datas:
            data, target = data.to(device), target.to(device)

            output = model(data)                                # 获得输出

            loss, iou, _ = criterion(output, target)            # 计算损失
            total_loss += loss.item()

            correct += iou.mean(dim=-1).mean(-1).sum().item()   # 计算正确率

            # 打印测试进度
            pbar.set_postfix(loss=loss.item())
            pbar.update(batch_size)

        pbar.set_postfix()                  # 清空进度条备注
        pbar.update(pbar.total - pbar.n)    # 防止进度条超过 100%

        average_loss = total_loss / dataset_batches
        accuracy = 100. * correct / dataset_size
        tqdm.write(
            f"Average loss: {average_loss:.4f}, "
            f"Accuracy: {correct:.0f}/{dataset_size} ({accuracy:.2f}%)")

    if (writer is not None) & (epoch is not None):
        writer.add_scalar('./logs/tensorboard/loss', average_loss, epoch)
        writer.add_scalar('./logs/tensorboard/accuracy', accuracy, epoch)

    return average_loss, accuracy


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
    elif config == "ResNet v2.0":
        _in_channels = image_shape[0]
        _dim_classifiers = [64, 16]
        _center_encoder = ResNeXtEncoder(
            _in_channels,
            num_blocks=[3, 4, 6, 3],
            dim_hidden=[32, 64, 128, 256])
        _size_encoder = ResNeXtEncoder(
            _in_channels,
            num_blocks=[3, 4, 6, 3],
            dim_hidden=[32, 64, 128, 256])

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


if __name__ == '__main__':
    # 超参数设置
    batch_size = 32
    num_epochs = 50
    num_workers = os.cpu_count()
    learning_rate = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    scheduler_step_size = 5
    scheduler_gamma = 0.5

    # _position = torch.randn(batch_size, 1, 2) * 32
    # _position_offset = torch.randn(batch_size, 1, 2) * 4
    # _size = torch.randn(batch_size, 1, 2) * 16 - 24
    # _offset_size = torch.randn(batch_size, 1, 2) * 64
    # for k in range(60):
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

    writer = get_writer('./')

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
    model = get_model(image_shape, "AlexNet")
    model.to(device)

    # 定义损失函数和优化器
    criterion = EllipseLoss(image_shape)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma)
    scaler = GradScaler()

    # 训练模型
    # 训练和验证模型，在 Terminal 激活 tensorboard 的指令:
    # tensorboard --logdir=./custom/EllipseDetectionNeuralNetwork/logs/tensorboard
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        train()
        _, accuracy = validate()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save.as_pt(model, "models/best.pt")
        scheduler.step()  # 更新学习率

    close_writer(writer)
