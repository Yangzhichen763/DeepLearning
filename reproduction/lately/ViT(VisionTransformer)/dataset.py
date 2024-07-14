import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np


class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)

        return {"image": image, "label": label, "index": index}


class MNISTValidationDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)

        return {"image": image, "label": label, "index": index}


class MNISTSubmissionDataset(Dataset):
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint)
        index = self.indices[idx]
        image = self.transform(image)

        return {"image": image, "index": index}
