import os.path

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from lately.segment_utils import *

from utils.os import *

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
            year="2012",
            image_set="train",
            download=False,
            transform_image=None,
            transform_label=None):
        root = os.path.join(get_root_path(), 'datas', 'VOCSegmentation')
        root = os.path.relpath(root)

        super(VOCSegmentationDataset, self).__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download)

        self.transform_image = transform_image
        self.transform_label = transform_label
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'potted plant', 'sheep', 'sofa', 'train',
            'tv/monitor']
        # 各种标签所对应的颜色
        self.color_map = torch.tensor(np.multiply([
            [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0], [0, 0, 2],
            [2, 0, 2], [0, 2, 2], [2, 2, 2], [1, 0, 0], [3, 0, 0],
            [1, 2, 0], [3, 2, 0], [1, 0, 2], [3, 0, 2], [1, 2, 2],
            [3, 2, 2], [0, 1, 0], [2, 1, 0], [0, 3, 0], [2, 3, 0],
            [0, 1, 2]], 64))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert('RGB')

        if self.transform_image is not None:
            image = self.transform_image(image)
        else:
            image = torch.tensor(image)

        if self.transform_label is not None:
            target = self.transform_label(target)
        else:
            target = torch.tensor(target)

        label = torch.zeros(target.shape[-2], target.shape[-1], dtype=torch.int64)
        for color in self.color_map:
            color = color.reshape(3, 1, 1)
            label += torch.eq(target, color).all(dim=0)

        return image, label


if __name__ == '__main__':
    _transform_image = get_transform(3)
    _transform_label = get_transform(1)

    _root = '../../datas/VOCSegmentation'
    _year = '2012'
    training_dataset = VOCSegmentationDataset(
        year=_year, image_set='train', download=True,
        transform_image=_transform_image, transform_label=_transform_label)

    training_loader = DataLoader(
        training_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4)

    for i, (images, targets) in enumerate(training_loader):
        print(images.shape, targets.shape)
        if i == 10:
            break
