import time

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import torchvision
import torchvision.transforms as transforms

from utils.os import get_root_path


class Checkpoint(pl.Callback):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.ckpt_path = ckpt_path

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        os.makedirs('models', exist_ok=True)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # 保存模型
        torch.save(model.state_dict(), self.ckpt_path or f'models/mlp_variant_{time.strftime("%Y-%m-%d-%H-%M-%S")}.pth')
        torch.save(model.state_dict(), f'models/mlp_variant.pth')


class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    # noinspection PyUnresolvedReferences
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)

        _, predicted = torch.max(y_hat.data, 1)

        correct = (predicted == y).sum().item()
        self.log('val_acc', correct, on_epoch=True)


if __name__ == '__main__':
    import torch.utils.data as data
    import pytorch_lightning.loggers as loggers

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # 加载训练集
    train_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, persistent_workers=True)
    # 加载测试集
    test_set = torchvision.datasets.MNIST(
        root=f'{get_root_path()}/datas/', train=False, download=True, transform=transform)
    val_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)

    # 定义模型
    model_kwargs = dict(input_size=784, hidden_size=128, output_size=10)
    model = MLP(**model_kwargs)

    ckpt_path = f'models/mlp_variant_{time.strftime("%Y-%m-%d-%H-%M-%S")}.ckpt'
    # 训练模型
    trainer = pl.Trainer(
        default_root_dir=f'models/',    # checkpoints 保存路径
        logger=loggers.TensorBoardLogger('logs/', name='mlp_variant'),    # 日志保存路径
        callbacks=[Checkpoint(ckpt_path)],
        max_epochs=50)
    trainer.fit(model, train_loader, val_loader)    # ckpt_path 是用来继续训练的

    # 测试模型
    mlp = MLP.load_from_checkpoint(checkpoint_path=ckpt_path, **model_kwargs)
    model.eval()
    trainer.test(model=mlp, dataloaders=val_loader)

