import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.os import get_root_path
from utils.general import Trainer, Manager, Validator


# noinspection PyUnresolvedReferences
def train(*, model, num_epochs,
          batch_size=64, learning_rate=0.001, num_workers=4, device=torch.device('cuda')):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # 加载训练集
    train_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 加载测试集
    test_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    trainer = Trainer(train_loader, optimizer)
    manager = Manager(model, device)
    validator = Validator(test_loader)
    for epoch in range(1, num_epochs + 1):
        train_datas = trainer.start(epoch, model=model)
        for (i, (x, y)) in train_datas:
            x = x.to(device)

            predict, loss = trainer.predict(model, x, y, criterion)  # 预测和计算损失
            trainer.backward(loss)  # 反向传播
            trainer.step(i, loss)   # 更新参数
        trainer.end()

        test_datas = validator.start(epoch, model=model, optimizer=optimizer)
        with torch.no_grad():
            for (i, (x, y)) in test_datas:
                x = x.to(device)

                logit, loss = validator.predict(model, x, y, criterion)  # 预测和计算损失

                _, predict = torch.max(logit, 1)
                num_correct = (predict == y).sum().item()
                validator.step(i, loss, num_correct)    # 记录准确率和损失
        average_loss, accuracy = validator.end()

        manager.update_checkpoint(accuracy)


# noinspection PyUnresolvedReferences
def test(model, x):
    tester = Tester(x, model=model)
    with torch.no_grad():
        x = x.to(device)

        logit, loss = tester.predict(model, x, y, criterion)  # 预测和计算损失

        # 计算准确率
        _, predict = torch.max(logit, 1)
        return predict[0]

