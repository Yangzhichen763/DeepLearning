# 用Pytorch实现：神经网络理论，使用损失函数和优化理论，在手写数字识别数据集MNIST和CIFAI-10上训练并得到分类精度
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import tqdm
from utils.os import root

from .model import MLP


def load_data(path=f'{root}/datas/MNIST/', batch_size=64, **loader_kwargs):
    if loader_kwargs is None:
        loader_kwargs = dict(
            num_workers=8,
            shuffle=True,
            persistent_workers=True,
        )

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root=f'{path}/train', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, **loader_kwargs)

    test_set = torchvision.datasets.MNIST(root=f'{path}/test', train=False, download=True, transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, **loader_kwargs)

    return train_loader, test_loader


def run():
    learning_rate = 3e-4
    batch_size = 64
    epochs = 50

    train_loader, test_loader = load_data(batch_size=batch_size)

    net = MLP(hidden_sizes=[256, 128])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # 训练网络
    for epoch in range(epochs):
        running_loss = 0.0

        datas = tqdm.tqdm(train_loader)
        for i, data in enumerate(datas, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            datas.set_description(f'Epoch {epoch + 1}/{epochs} loss: {running_loss / (i + 1):.4f}')

        datas.set_description(f'Epoch {epoch + 1}/{epochs} loss: {running_loss / len(train_loader):.4f}')

    # 测试网络
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # noinspection PyUnresolvedReferences
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    run()
