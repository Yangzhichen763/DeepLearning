"""
AABB (Axis-Aligned Bounding Box)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F





def soft_iou(pred, target, eps=1e-9, theta=4):
    """
    传入的 pred、target 应该是 torch.tensor([x, y, w, h])
    论文链接：https://arxiv.org/abs/2205.12740
    Args:
        pred (torch.Tensor):
        target (torch.Tensor):
        theta (float): 可以在 2~6 之间取值，如果 theta 的值为 1，则会立即优化形状，从而损害形状的自由运动
        eps (float): 防止除零错误

    Returns:

    """
    cx_p, cy_p, w_p, h_p = pred.split(1, dim=-1)
    cx_t, cy_t, w_t, h_t = target.split(1, dim=-1)

    c_h = torch.abs(cy_t - cy_p)  # 中心之间的竖直距离
    c_w = torch.abs(cx_t - cx_p)  # 中心之间的水平距离

    def _angle_cost():
        _sigma = torch.sqrt((cx_t - cx_p) ** 2 + (cy_t - cy_p) ** 2)
        _alpha = torch.arcsin(c_h / _sigma)
        _lambda = torch.sin(2 * _alpha)
        return _lambda

    def _distant_cost(_lambda):
        _gamma = 2 - _lambda
        _rho_x = ((cx_t - cx_p) / c_w) ** 2
        _rho_y = ((cy_t - cy_p) / c_h) ** 2
        _delta = (1 - torch.exp(-_gamma * _rho_x)) + (1 - torch.exp(-_gamma * _rho_y))
        return _delta

    def _shape_cost():
        _omega_w = torch.abs(w_t - w_p) / torch.max(w_t, w_p)
        _omega_h = torch.abs(h_t - h_p) / torch.max(h_t, h_p)
        _omega = (1 - torch.exp(-_omega_w)) ** theta + (1 - torch.exp(-_omega_h)) ** theta
        return _omega

    angle_cost = _angle_cost()
    distant_cost = _distant_cost(angle_cost)
    shape_cost = _shape_cost()
    return 1 - iou(pred, target, eps) + (distant_cost + shape_cost) / 2


def iou(pred, target, eps=1e-9):
    """
    传入的 pred、target 应该是 torch.tensor([x, y, w, h])
    Args:
        pred (torch.Tensor):
        target (torch.Tensor):
        eps (float): 防止除零错误

    Returns:

    """
    boxes = torch.cat([pred, target]).permute(1, 0)
    x, y, w, h = boxes
    x_min, y_min = x - w / 2.0, y - h / 2.0
    x_max, y_max = x + w / 2.0, y + h / 2.0

    # 获取矩形框交集对应的左上角和右下角的坐标
    x1, y1 = torch.max(x_min), torch.max(y_min)
    x2, y2 = torch.min(x_max), torch.min(y_max)

    # 计算两个矩形框面积
    area = (x_max - x_min) * (y_max - y_min)
    inter_area = (torch.clamp_min(x2 - x1, 0)) * (torch.clamp_min(y2 - y1, 0))  # 计算交集面积
    union_area = torch.sum(area) - inter_area  # 计算并集面积
    iou = inter_area / (union_area + eps)  # 计算交并比

    return iou
