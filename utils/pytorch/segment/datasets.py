
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from PIL import Image

from utils.os import *
import os.path

from utils.pytorch.segment import save_label_as_png


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
    # 各种标签所对应的颜色
    color_map = torch.mul(torch.tensor([
        [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0], [0, 0, 2],
        [2, 0, 2], [0, 2, 2], [2, 2, 2], [1, 0, 0], [3, 0, 0],
        [1, 2, 0], [3, 2, 0], [1, 0, 2], [3, 0, 2], [1, 2, 2],
        [3, 2, 2], [0, 1, 0], [2, 1, 0], [0, 3, 0], [2, 3, 0],
        [0, 1, 2]]), 64)
    classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'potted plant', 'sheep', 'sofa', 'train',
        'tv/monitor']

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

        if transform_image is None:
            transform_image = transforms.ToTensor()
        if transform_label is None:
            transform_label = transforms.ToTensor()
        self.transform_image = transform_image
        self.transform_label = transform_label

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns: image: [3, H, W], label: [H, W]

        """
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert('RGB')

        image = self.transform_image(image)
        target = self.transform_label(target)

        if not (target > 1).any():
            target = (target * 255.0).long()
        label = torch.zeros(target.shape[-2], target.shape[-1]).long()
        for (i, color) in enumerate(self.color_map.squeeze()):
            color = color.reshape(3, 1, 1)
            label += i * torch.eq(target, color).all(dim=0)  # target: [3, H, W], color: [3, 1, 1]

        return image, label     # [3, H, W], [H, W]


class CarvanaDataset(Dataset):
    color_map = torch.tensor([[0, 0, 0], [255, 255, 255]])
    classes = ['background', 'car']

    def __init__(
            self,
            image_dir_name="images",
            mask_dir_name="masks",
            transform_image=None,
            transform_label=None):
        root = os.path.join(get_root_path(), 'datas', 'Carvana', 'train')
        root = os.path.relpath(root)
        self.image_dir = os.path.join(root, image_dir_name)
        self.mask_dir = os.path.join(root, mask_dir_name)

        if transform_image is None:
            transform_image = transforms.ToTensor()
        if transform_label is None:
            transform_label = transforms.ToTensor()
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_image(image)
        mask = self.transform_label(mask)

        if not (mask > 1).any():
            mask = mask / 255.0
        label = mask.long()

        return image, label


def get_transform(channels=3):
    """
    获取数据预处理器
    :return:
    """
    if channels == 3:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    elif channels == 1:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])
    else:
        raise ValueError(f"Unsupported channels: {channels}")

    return transform


if __name__ == '__main__':
    # 测试 CarvanaDataset
    carvana_dataset = CarvanaDataset(
        transform_image=get_transform(3),
        transform_label=get_transform(1)
    )
    carvana_loader = DataLoader(
        carvana_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4)

    for (_i, (images, targets)) in enumerate(carvana_loader):
        print(images.shape, targets.shape)
        save_label_as_png(
            targets,
            device='cpu',
            color_map=CarvanaDataset.color_map,
            dir_path='./logs/labels',
            file_name=f"carvana_label_{_i}")
        if _i == 10:
            break

    exit()
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

    for (_i, (images, targets)) in enumerate(training_loader):
        print(images.shape, targets.shape)
        if _i == 10:
            break
