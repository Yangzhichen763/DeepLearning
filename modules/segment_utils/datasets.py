import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from PIL import Image

import os

from torchvision.io import read_image


def read_images(image_path, resize_and_crop_transformer=None):
    if resize_and_crop_transformer is None:
        resize_and_crop_transformer = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
            ])

    # 读取图片
    display_images = []
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(image_path, filename)
            # 读取图片
            image = read_image(path)
            # 调整图片大小并裁剪中心区域
            image = resize_and_crop_transformer(image)
            display_images.append(image)


class VOCSegmentationDataset(torchvision.datasets.VOCSegmentation):
    def __init__(
            self,
            root,
            year="2012",
            image_set="train",
            download=False,
            transform=None):
        """

        Args:
            root (str):
            year (str):
            image_set (str):
            download (bool):
            transform:
        """
        super(VOCSegmentationDataset, self).__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform)
        self.transform = transform
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'potted plant',
            'sheep', 'sofa', 'train', 'tv/monitor']
        # 各种标签所对应的颜色
        self.colormap = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0]]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert('RGB')

        if self.transform is not None:
            image, target = self.transform(image), self.transform(target)

        return image, target


if __name__ == '__main__':
    _transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.ToPILImage(),
        ])

    _root = '../../datas/VOCSegmentation'
    _year = '2012'
    training_dataset = VOCSegmentationDataset(
        root=_root, year=_year, image_set='train', download=True, transform=_transform)

    training_loader = DataLoader(
        training_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4)

    for _data in training_loader:
        images, targets = _data
        print(images.shape, targets.shape)
