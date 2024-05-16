import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd

from utils.os import get_root_path


class EllipseDetectionDataset(Dataset):
    def __init__(
            self,
            data_set="1x",
            image_dir_name="images",
            label_dir_name="labels"):
        """

        Args:
            data_set: 可以选择 "1x"、"2x"或"5x"
            image_dir_name:
            label_dir_name:
        """
        print("Loading EllipseDetectionDataset...")
        root = os.path.join(get_root_path(), 'datas', 'EllipseDetection', 'train', f'{data_set}')
        root = os.path.relpath(root)
        self.image_dir = os.path.join(root, image_dir_name)
        self.label_dir = os.path.join(root, label_dir_name)

        self.label_length = int(data_set[:data_set.index('x')])
        label_csv_file = os.path.join(self.label_dir, 'labels.csv')
        self.labels = pd.read_csv(label_csv_file, sep=',', header=None)
        print(f"EllipseDetectionDataset loaded, {int(len(self.labels) / self.label_length)} labels found.")

    def __getitem__(self, index):
        row = self.labels.values[index * self.label_length: (index + 1) * self.label_length, :]
        image_file_name = row[0, 0]
        rotated_rect = row[:, 2:].astype(np.float32)
        label = torch.tensor(rotated_rect).permute(1, 0)

        image_path = os.path.join(self.image_dir, f'{image_file_name}.jpg')
        image = Image.open(image_path).convert("L")
        image = transforms.ToTensor()(image)
        return image, label

    def __len__(self):
        return len(self.labels) // self.label_length


if __name__ == '__main__':
    dataset = EllipseDetectionDataset(data_set="1x")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for (i, (_image, _label)) in enumerate(dataloader):
        print(_image.shape, _label.shape, _label)
        if i == 10:
            break
