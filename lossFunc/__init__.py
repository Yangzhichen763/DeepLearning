import torch
import torch.nn as nn

from lossFunc.BBLoss import WHRatioLoss, FocalLoss


class LogLoss(nn.Module):
    def __init__(self, eps=1e-15):
        super(LogLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        return loss.mean()
