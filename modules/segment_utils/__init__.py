import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils.pytorch import *
from utils.pytorch.segment import *

import os


def get_transform():
    """
    获取数据预处理器
    :return:
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transform


def get_data_set(transform):
    """
    获取数据集
    :param transform: 数据预处理器
    :return: 训练集和测试集
    """
    # 加载 CIFAR-10 数据集
    root = '../../datas/VOCSegmentation'
    year = '2012'
    training_dataset = torchvision.datasets.VOCSegmentation(
        root=root, year=year, image_set='train', download=True, transform=transform)    # 训练集
    test_dataset = torchvision.datasets.VOCSegmentation(
        root=root, year=year, image_set='val', download=True, transform=transform)      # 测试集
    return training_dataset, test_dataset


def get_data_loader(datasets, batch_size, num_samples, shuffle=True, num_workers=4):
    """
    获取数据加载器
    :param datasets: 数据集（包括训练集和验证集）
    :param batch_size: 批大小
    :param num_samples: 训练集和测试集样本数量，比如 [5000, 1000]
    :param shuffle: 是否打乱数据
    :param num_workers: 加载数据进程数
    :return: 数据加载器
    """
    training_dataset, test_dataset = datasets
    training_num_samples, test_num_samples = num_samples

    training_subset = pick_random(training_dataset, training_num_samples, "training")
    test_subset = pick_random(test_dataset, test_num_samples, "test")

    # 定义数据加载器
    training_loader = DataLoader(training_subset, batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size, shuffle=shuffle, num_workers=num_workers)
    return training_loader, test_loader


def train_model(model, optimizer, criterion, scheduler, loaders, device='cuda', num_epochs=10, writer=None):
    """
    训练模型
    Args:
        model: 待训练的模型
        optimizer: 优化器
        criterion: 损失函数
        scheduler: 学习率调度器
        loaders: 数据加载器，包括训练集和测试集的加载器
        device: 设备类型
        writer: 日志记录器
        num_epochs: 训练轮数

    Returns:
        训练好的模型
    """
    training_loader, test_loader = loaders

    # 训练模型
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        print('Training Epoch: %d' % epoch)
        train_model_in_single_epoch(model, training_loader, optimizer, criterion, device, writer, epoch)
        _, accuracy = validate_model(model, test_loader, criterion, device, writer, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save.as_pt(model, "./models/best.pt")
        scheduler.step()  # 更新学习率


def train_and_validate(
        transform,
        model_creator,

        batch_size,
        num_samples,

        learning_rate=0.001,
        weight_decay=1e-4,
        momentum=0.9,
        scheduler_step_size=2,
        scheduler_gamma=0.5,

        device='cuda',
        num_epochs=10,
        writer=None,
):
    """
    训练并验证模型
    Args:
        transform: 数据预处理器
        model_creator: 模型创建器，接收 num_classes 作为参数，返回模型
        batch_size: 批大小
        num_samples: 训练集和测试集样本数量，比如 [5000, 1000]
        learning_rate: 学习率
        weight_decay: 权重衰减
        momentum: 动量
        scheduler_step_size: 学习率更新步长
        scheduler_gamma: 学习率衰减率，每个步长衰减学习率为 gamma * 学习率
        device: 设备类型
        num_epochs: 训练轮数
        writer: 日志记录器

    Returns:
        模型的类别数量
    """
    # ------------------------ 数据集处理部分
    # 获取数据集
    datasets = get_data_set(transform)
    # 获取数据加载器
    loaders = get_data_loader(
        datasets=datasets,
        batch_size=batch_size,
        num_samples=num_samples)

    # ------------------------ 模型创建部分
    # 获取数据集中类别数量
    training_dataset, _ = datasets
    num_classes = 21  # 2012年VOC数据集类别数量为 21（包括背景）
    # 创建模型
    model = model_creator(num_classes=num_classes).to(device)

    # ------------------------ 训练和测试部分
    optimizer = torch.optim.SGD(                    # 定义优化器
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()         # 定义损失函数
    scheduler = torch.optim.lr_scheduler.StepLR(    # 定义学习率调度器
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma)
    # 训练模型
    train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        loaders=loaders,
        device=device,
        num_epochs=num_epochs,
        writer=writer)

    # 保存模型
    save.as_pt(
        model,
        f"./models/{model_creator.__name__}.pt")

    return num_classes