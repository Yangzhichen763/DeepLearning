import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
from PIL import Image

import math

from custom.EllipseDetectionNeuralNetwork.loss import EllipseLoss
from modules.residual.ResNet import resnet18
from optim import CosineAnnealingWarmupRestarts

learning_rate = 1e-2
num_epochs = 100

model = resnet18()

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate)
scheduler_epoch = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=int(num_epochs / 10),
    max_lr=learning_rate,
    min_lr=1e-8,
    warmup_steps=5,
    gamma=0.1)

for epoch in range(1, num_epochs + 1):
    scheduler_epoch.step()
    print(f"Epoch {epoch}: learning rate = {optimizer.param_groups[0]['lr']}")
