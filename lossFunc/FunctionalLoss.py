import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Functional01Loss(nn.Module):
    """
    越接近 0，损失越低；越接近 1，损失越高。
    """
    def __init__(self):
        super(Functional01Loss, self).__init__()

    def forward(self, pred):
        return self._loss(pred)


class L2Loss(Functional01Loss):
    """
    loss = pred ^ 2
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def _loss(self, pred):
        return F.mse_loss(pred, torch.zeros_like(pred))


class L1Loss(Functional01Loss):
    """
    loss = |pred|
    """
    def __init__(self):
        super(L1Loss, self).__init__()

    def _loss(self, pred):
        return F.l1_loss(pred, torch.zeros_like(pred))  # 或者 torch.abs(pred)


class SmoothL1Loss(Functional01Loss):
    """
    loss = 0.5 * |pred|^2 if |pred| < 1
    loss = |pred| - 0.5 if |pred| >= 1
    """
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def _loss(self, pred):
        return F.smooth_l1_loss(pred, torch.zeros_like(pred))


class LogLoss(Functional01Loss):
    """
    loss = -log(1 - pred) - log(1 + pred)
    """
    def __init__(self):
        super(LogLoss, self).__init__()

    def _loss(self, pred):
        return -torch.log(1 - pred) - torch.log(1 + pred)


class CoshLoss(Functional01Loss):
    """
    loss = 2 * (cosh(alpha * pred) - 1)
    """
    def __init__(self, alpha=1.0):
        super(CoshLoss, self).__init__()
        self.alpha = alpha

    def _loss(self, pred):
        return 2 * (torch.cosh(self.alpha * pred) - 1)


class SquareExpLoss(Functional01Loss):
    """
    loss = exp(alpha * pred^2) - 1
    """
    def __init__(self, alpha=1.0):
        super(SquareExpLoss, self).__init__()
        self.alpha = alpha

    def _loss(self, pred):
        return torch.exp(self.alpha * pred ** 2) - 1


class DSigmoidLoss(Functional01Loss):
    """
    loss = 1 - 4 * exp(-alpha * pred) / ((1 + exp(-alpha * pred))^2)
    """
    def __init__(self, alpha=4.0):
        super(DSigmoidLoss, self).__init__()
        self.alpha = alpha

    def _loss(self, pred):
        exp_pred = torch.exp(-self.alpha * pred)
        return 1 - 4 * exp_pred / ((1 + exp_pred) ** 2)


class SquareArctanLoss(Functional01Loss):
    def __init__(self, alpha=4.0):
        super(SquareArctanLoss, self).__init__()
        self.alpha = alpha

    def _loss(self, pred):
        return 2 * torch.arctan((self.alpha * pred) ** 2) / math.pi
