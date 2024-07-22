import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm

from reproduction.lately.ScoreSDEs.modules import ScoreUNet
from utils.torch import deploy
from utils.general import Trainer, Manager

from model import ScoreSDETrainer, marginal_prob_std

device = deploy.assert_on_cuda()

score_model = torch.nn.DataParallel(ScoreUNet(marginal_prob_std=marginal_prob_std))
score_model = score_model.to(device)

num_epochs = 50
batch_size = 32
learning_rate = 1e-4

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
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

        predict, loss = trainer.predict(sde_model, x, y, sde_model.loss_func)  # 预测和计算损失
        trainer.backward(loss)  # 反向传播
        trainer.step(i, loss)   # 更新参数

    trainer.end()
    manager.update_checkpoint(0.)
