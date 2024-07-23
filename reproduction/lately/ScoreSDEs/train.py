import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from parts import ScoreUNet
from utils.torch import deploy
from utils.general import Trainer, Manager
from utils.os import get_root_path
from utils.log.info import print_

from model import ScoreSDETrainer, marginal_prob_std

if __name__ == '__main__':
    device = deploy.assert_on_cuda()

    score_model = torch.nn.DataParallel(ScoreUNet(marginal_prob_std=marginal_prob_std))
    score_model = score_model.to(device)

    num_epochs = 50
    batch_size = 32
    learning_rate = 1e-4

    dataset = MNIST(get_root_path('datas/MNIST'), train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    sde_model = ScoreSDETrainer(score_model)
    optimizer = Adam(score_model.parameters(), lr=learning_rate)

    trainer = Trainer(data_loader, optimizer)
    manager = Manager(score_model, device)
    for epoch in range(1, num_epochs + 1):
        datas = trainer.start(epoch)
        sde_model.train()
        for (i, (x, y)) in datas:
            x = x.to(device)

            loss = sde_model.loss_func(x)  # 计算损失

            trainer.backward(loss)  # 反向传播
            trainer.step(i, loss)   # 更新参数

        trainer.end()
        manager.update_checkpoint(0.)
