import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from custom.EllipseDetectionNeuralNetwork.datasets import EllipseDetectionDataset
from utils.tensorboard import *
from utils.pytorch import *
from utils.pytorch.dataset import *
from utils.os import *

from model import EllipseDetectionNetwork
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
        box = box.detach()
        box = torch.clamp_min(box, 1)
        box = box.cpu().numpy()
        box = (box[0:2], box[2:4], box[4])
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
    for i, (prrs, trrs, ious) in enumerate(zip(pred, target, ioubs)):
        for trr in trrs:
            _draw_ellipse(images[i], trr,
                          color=(64, 64, 64),
                          thickness=12)
        for j, (prr, iou) in enumerate(zip(prrs, ious)):
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

    model.train()  # 设置模型为训练模式
    with tqdm(
            total=dataset_size,
            unit='image') as pbar:
        for (i, (images, target)) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()                           # 清空梯度
            output = model(images)                          # 前向传播，获得输出
            loss, iou, pred = criterion(output, target)     # 计算损失
            loss.backward()                                 # 反向传播
            optimizer.step()                                # 更新参数

            # if epoch >= 5:
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

    if epoch is not None:
        tqdm.write(
            f"Epoch {epoch} training finished. "
            f"Min loss: {min_loss:.6f}, "
            f"Max loss: {max_loss:.6f}, "
            f"Avg loss: {total_loss / len(train_loader):.6f}")


def validate():
    dataset_size = len(test_loader.dataset)
    dataset_batches = len(test_loader)

    total_loss = 0
    correct = 0
    model.eval()
    with tqdm(
            total=dataset_size,
            unit='image') as pbar, torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)                            # 前向传播，获得输出
            loss, iou, _ = criterion(output, target)        # 计算损失

            total_loss += loss.item()
            correct += iou.mean(dim=-1).mean(-1).sum().item()        # 计算正确率

            # 打印测试进度
            pbar.set_postfix(loss=loss.item())
            pbar.update(batch_size)

    average_loss = total_loss / dataset_batches
    accuracy = 100. * correct / dataset_size
    tqdm.write(
        f"\nTest set: Average loss: {average_loss:.4f}, Accuracy: {correct:.00f}/{dataset_size} "
        f"({accuracy:.00f}%)")

    if (writer is not None) & (epoch is not None):
        writer.add_scalar('./logs/tensorboard/loss', average_loss, epoch)
        writer.add_scalar('./logs/tensorboard/accuracy', accuracy, epoch)

    return average_loss, accuracy


if __name__ == '__main__':
    # 超参数设置
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-5
    weight_decay = 0.0001
    momentum = 0.9
    scheduler_step_size = 2
    scheduler_gamma = 0.5

    # _pred = torch.randn(batch_size, 64, 5) * 8
    # _target = EllipseLoss.regress_rr(torch.randn(batch_size, 2, 5) * 8, (256, 256))
    # _target_image = torch.zeros((batch_size, 1, 256, 256))
    # _loss, _iou, _pred = EllipseLoss((256, 256))(_pred, _target)
    # save_image(_pred, _target, _target_image, 0, _iou)
    # exit()

    # 部署 GPU 设备
    device = assert_on_cuda()

    writer = get_writer('./')

    # 加载数据集
    dataset = EllipseDetectionDataset(data_set="1x")
    train_set, val_set = datapicker.split_train_test(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义模型
    image_shape = (1, 256, 256)
    model = EllipseDetectionNetwork(
        input_shape=image_shape,
        dim_features=[16, 32, 64, 128, 256],
        num_layers_in_features=[2, 2, 3, 3, 2],
        dim_classifiers=[64, 32],
        device=device
    )
    model.to(device)

    # 定义损失函数和优化器
    criterion = EllipseLoss(image_shape)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=learning_rate,
    #     weight_decay=weight_decay,
    #     momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma)

    # 训练模型
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        print('Training Epoch: %d' % epoch)
        train()
        _, accuracy = validate()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save.as_pt(model, "./models/best.pt")
        scheduler.step()  # 更新学习率

    close_writer(writer)
