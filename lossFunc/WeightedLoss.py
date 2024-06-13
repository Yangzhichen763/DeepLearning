import torch
import torch.nn as nn
from torch.nn import functional as F


class WeightedLoss(nn.Module):
    def forward(self, pred, target, weights=1.0):
        loss = self._loss(pred, target)
        weighted_loss = loss * weights
        return weighted_loss


class WeightedL1Loss(WeightedLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)


class WeightedL2Loss(WeightedLoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='none')
