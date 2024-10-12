import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from utils.os import get_root_path
from utils.general import Trainer, Validator, Manager
from testime.mnist import train, test


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Model(nn.Module):
    def __init__(self, *, backbone, in_features=512, hidden_size=256, num_classes):
        super(Model, self).__init__()
        self.backbone = backbone
        self.fc = MLP(in_features, hidden_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_with_embeddings(self, x):
        embedding = self.backbone(x)
        embedding = embedding.view(x.size(0), -1)
        x = self.fc(embedding)
        return embedding, x


# noinspection PyUnresolvedReferences
def train(*, model, num_epochs,
          batch_size=64, learning_rate=0.001, num_workers=4, device=torch.device('cuda')):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载训练集
    train_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 加载测试集
    test_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 定义损失函数和优化器
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化模型训练工具
    trainer = Trainer(train_loader, optimizer)
    manager = Manager(model, device)
    validator = Validator(test_loader)
    def preprocess(x):
        x = x.to(device).repeat(1, 3, 1, 1)
        return x

    # 初始化标签 embedding
    embeddings = [None] * 10
    for image_batch, label_batch in train_loader:
        for image, label in zip(image_batch, label_batch):
            image = preprocess(image)
            label = label.item()

            if embeddings[label] is None:
                embeddings[label] = model(image.repeat(batch_size, 1, 1, 1))[0]
                print(embeddings[label].shape)

                if all(x is not None for x in embeddings):
                    break

    # 1. 空间中相邻的两个点标签应该是相同的

    # 分类训练
    for epoch in range(1, num_epochs + 1):
        train_datas = trainer.start(epoch, model=model)
        for (i, (x, _)) in train_datas:  # 半监督，所以标签不参与训练
            x = preprocess(x)

            embedding, predict = model.forward_with_embedding(x)
            loss = 0
            for j in range(10):
                if j == predict:
                    continue
                loss += criterion(embedding, embeddings[predict], embeddings[j])

            trainer.backward(loss)  # 反向传播
            trainer.step(i, loss)   # 更新参数
        trainer.end()

    # 聚类训练
    embeddings = [[]] * 10
    embedding_centroids = [None] * 10   # 存储每个类别 embedding 的中心，防止重复计算
    for epoch in range(1, num_epochs + 1):
        train_datas = trainer.start(epoch, model=model)
        for (i, (x, _)) in train_datas:  # 半监督，所以标签不参与训练
            x = preprocess(x)

            embedding, predict = model.forward_with_embedding(x)
            embeddings[predict].append(embedding)
            embedding_centroids[predict] = torch.mean(torch.stack(embeddings[predict]), dim=0)

            loss = 0
            for j in range(10):
                if j == predict or embedding_centroids[j] is None:
                    continue
                loss += criterion(embedding, embedding_centroids[predict], embedding_centroids[j])

            trainer.backward(loss)  # 反向传播
            trainer.step(i, loss)  # 更新参数
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


def get_model():
    # 修改 torch.hub 下载模型的路径
    import os
    os.environ["TORCH_HOME"] = "models/"
    torch.hub._download_url_to_file = "models/"

    # 加载 ResNet-50 主干网络
    backbone = resnet50(pretrained=True)
    # 冻结参数
    for param in backbone.parameters():
        param.requires_grad = False

    # 定义模型
    model = Model(backbone=backbone, in_features=2048, hidden_size=256, num_classes=10)
    return model


if __name__ == '__main__':
    # 超参数设置
    batch_size = 64
    num_workers = 2
    num_epochs = 10
    learning_rate = 3e-4
    input_size = (28, 28)
    hidden_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练模型
    _model = get_model()
    train(model=_model, num_epochs=num_epochs,
          batch_size=batch_size, learning_rate=learning_rate, device=device, num_workers=num_workers)

