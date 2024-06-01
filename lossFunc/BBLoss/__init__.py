import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lossFunc.BBLoss.AABB import *
from lossFunc.BBLoss.OBB import *


class WHRatioLoss(nn.Module):
    """
    旋转框的宽高比损失函数（如果将中心重合，宽高比之间的比值越接近 1，并且旋转角度之差越小，则损失越小）
    """
    def __init__(self, loss_func=None):
        super(WHRatioLoss, self).__init__()
        if loss_func is None:
            loss_func = self.linear_regression  # 最好使用 linear，否则容易对宽高比为 1 的框过拟合
        self.loss_func = loss_func

    def forward(self, x, y, a_x, a_y):
        """
        计算两个宽高比（W/H）之间的损失，公式如下：
        loss = (2 * (x.arctan() - y.arctan() + rotation).sin().asin() / math.pi) ** 2
        其中 x, y 为宽高比，rotation 为旋转角度，loss 为损失值。
        Args:
            x: 包围盒A的宽高比，shape=[B, N, 1]
            y: 包围盒B的宽高比，shape=[B, N, 1]
            a_x: 包围盒A的旋转角度，shape=[B, N, 1]
            a_y: 包围盒B的旋转角度，shape=[B, N, 1]

        Returns:

        """
        score = self.arctan_regression(x, y, a_x, a_y)  # [B, N, 1]
        score_log = self.loss_func(score)               # [B, N, 1]
        loss = score_log.sum(dim=-1)                    # [B, N]
        return loss, score_log                          # [B, N], [B, N, 1]

    @staticmethod
    def arctan_regression(x, y, a_x, a_y):
        cos_a = (a_x - a_y).cos()
        sin_a = (a_x - a_y).sin()
        cos_r = (x.arctan() + y.arctan()).cos()
        sin_r = (x.arctan() - y.arctan()).sin()
        score = (sin_r * cos_a + cos_r * sin_a).arcsin() * (2 / math.pi)
        return score

    @staticmethod
    def log_regression(x):
        return -(1 + x).log() - (1 - x).log()

    @staticmethod
    def exp_regression(x):
        return x.exp() + (-x).exp() - 2

    @staticmethod
    def square_regression(x):
        return x ** 2

    @staticmethod
    def linear_regression(x):
        return x.abs()


class DirectionLoss(nn.Module):
    """
    旋转框主轴所指方向损失函数
    """
    def __init__(self):
        super(DirectionLoss, self).__init__()

    def forward(self, wha1, wha2):
        pass


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=-1)
        target = F.one_hot(target, num_classes=pred.shape[-1]).float()
        print(pred, target)

        fl_p = -(1 - pred) ** self.gamma * torch.log(pred)

        loss = fl_p * target
        loss = loss.sum(dim=1)
        return loss


def get_covariance_matrix(rotated_rects):
    """
    计算旋转矩形的协方差矩阵
    Args:
        rotated_rects (torch.Tensor): torch.tensor([x, y, w, h, angle])，其中 angle 为弧度制

    Returns:
        (torch.Tensor): rotated_rects 的协方差矩阵。
    """
    # 计算 Gaussian Bounding Boxes
    wh, angle = rotated_rects[..., 2:4], rotated_rects[..., 4:]     # 不能写成 [..., 4]，提取出来的张量会少一维
    gbbs = torch.cat((wh.pow(2) / 12, angle), dim=-1)
    w, h, a = gbbs.split(1, dim=-1)
    cos_a, sin_a = a.cos(), a.sin()
    cos2_a, sin2_a = cos_a ** 2, sin_a ** 2
    return w * cos2_a + h * sin2_a, w * sin2_a + h * cos2_a, (w - h) * cos_a * sin_a


if __name__ == '__main__':
    _input_shape = (2, 3, 8)
    _pred = torch.randn(_input_shape)
    _target = torch.randint(0, _input_shape[-1], (*_input_shape[:-1],))
    _loss = FocalLoss(gamma=0.5)
    print(_loss(_pred, _target))
