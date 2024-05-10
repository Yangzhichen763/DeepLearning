import torch.utils.data
from torch import clamp
from torch.utils.data import Dataset, Subset, random_split
import numpy as np
from typing import Sequence, TypeVar

T_co = TypeVar('T_co', covariant=True)


def pick_sequential(dataset, num_samples):
    """
    从 dataset 数据集中按顺序选取 num_samples 个样本。
    Args:
        dataset (Dataset): 数据集
        num_samples (int): 选取样本数量
    """
    if num_samples > len(dataset):
        num_samples = len(dataset)
    return Subset(dataset, np.arange(num_samples))


def pick_range_sequential(dataset, indices):
    """
    从 dataset 数据集中按顺序选取 indices 指定的样本。
    Args:
        dataset (Dataset): 数据集
        indices (Sequence[int]): 选取样本的索引
    """
    return Subset(dataset, indices)


def pick_random(data, num_samples, dataset_name=None):
    """
    从 data 数据集中随机选取 num_samples 个样本。
    Args:
        data (Dataset): 数据集
        num_samples (int): 选取样本数量
        dataset_name (str): 数据集名称
    """
    if num_samples < 0 or num_samples > len(data):
        num_samples = len(data)
    # 从原始数据集中随机选择一定数量的样本索引

    print(f'Samples from the {dataset_name} set: {num_samples}/{len(data)}')
    subset_indices = torch.randperm(len(data))[:num_samples]
    return Subset(data, subset_indices)


# -------------------------- 以下为数据集划分函数 --------------------------

def split_train_test(dataset, test_ratio=0.2):
    """
    将数据集划分为训练集和测试集。
    Args:
        dataset (Dataset): 数据集
        test_ratio (float): 测试集占比
    """
    # 划分训练集和测试集
    train_size = int(len(dataset) * (1 - test_ratio))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def split_train_validation_test(dataset, lengths):
    """
    将数据集划分为训练集、验证集和测试集。
    Args:
        dataset (Dataset): 数据集
        lengths (Sequence[int | float]): 训练集、验证集、测试集的比例，如 [0.6, 0.2, 0.2]
    """
    # 划分训练集和测试集
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths)
    return train_dataset, validation_dataset, test_dataset

