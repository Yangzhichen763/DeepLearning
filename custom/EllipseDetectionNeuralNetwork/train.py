import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom.EllipseDetectionNeuralNetwork.datasets import EllipseDetectionDataset
from utils.tensorboard import *
from utils.pytorch import *
from utils.pytorch.dataset import *

from model import EllipseDetectionNetwork
from loss import *


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
        for (i, (data, target)) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()               # 清空梯度
            output = model(data)                # 前向传播，获得输出
            print(output.shape, target.shape)
            loss = obbox_iou(output, target, iou_type="DIoU")      # 计算损失
            print(loss.item())
            loss.backward()                     # 反向传播
            optimizer.step()                    # 更新参数

            # 记录损失最小值和最大值以及总损失
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())
            total_loss += loss.item()

            # 打印训练进度
            i_current_batch = i + 1
            pbar.update(batch_size)

            if writer is not None:
                writer.add_scalar('./logs/tensorboard/loss', loss.item(), i_current_batch)

    if epoch is not None:
        print(f"Epoch {epoch} training finished. "
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
            output = model(data)                                                # 前向传播，获得输出
            loss = criterion(output, target)                                    # 计算损失
            iou = obbox_iou(output, target, iou_type="GIoU").argmax(dim=1, keepdim=True)                     # 预测类别

            total_loss += loss.item()
            correct += iou.item()   # 计算正确率

            # 打印测试进度
            pbar.update(batch_size)

    average_loss = total_loss / dataset_batches
    accuracy = 100. * correct / dataset_size
    print(
        f"\nTest set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{dataset_size} "
        f"({accuracy:.0f}%)\n")

    if (writer is not None) & (epoch is not None):
        writer.add_scalar('./logs/tensorboard/loss', average_loss, epoch)
        writer.add_scalar('./logs/tensorboard/accuracy', accuracy, epoch)

    return average_loss, accuracy


if __name__ == '__main__':
    # 超参数设置
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    scheduler_step_size = 2
    scheduler_gamma = 0.5

    # 部署 GPU 设备
    device = assert_on_cuda()

    writer = get_writer('./')

    # 加载数据集
    dataset = EllipseDetectionDataset(data_set="1x")
    train_set, val_set = datapicker.split_train_test(dataset, [0.8, 0.2])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义模型
    model = EllipseDetectionNetwork(
        input_shape=(1, 256, 256),
        dim_features=[16, 32, 64, 128, 256],
        num_layers_in_features=[2, 2, 3, 3, 2],
        dim_classifiers=[64, 32],
        device=device
    )
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
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
