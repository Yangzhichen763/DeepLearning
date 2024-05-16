import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


class EllipseLoss(nn.Module):
    def __init__(self):
        super(EllipseLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # pred: [batch_size, 5, 64]
        # target: [batch_size, 5, 1]
        # 0,1 是椭圆中心坐标
        # 2,3 是椭圆长短轴长度
        # 4   是椭圆旋转角度

        iou = obbox_iou(pred, target, iou_type='GIoU').squeeze()
        loss_box = (1.0 - iou).mean()
        print(loss_box)

        # iou
        loss = self.loss_func(iou, torch.zeros_like(iou).long())
        return loss



def AABB_iou(box_1, box_2, box_minmax=False):
    boxes = torch.tensor([box_1, box_2]).permute(1, 0)
    if box_minmax:
        x_min, y_min, x_max, y_max = boxes
    else:
        x, y, w, h = boxes
        x_min, y_min = x - w / 2.0, y - h / 2.0
        x_max, y_max = x + w / 2.0, y + h / 2.0
    # 获取矩形框交集对应的左上角和右下角的坐标
    x1, y1 = torch.max(x_min), torch.max(y_min)
    x2, y2 = torch.min(x_max), torch.min(y_max)
    # 计算两个矩形框面积
    area = (x_max - x_min) * (y_max - y_min)
    inter_area = (torch.clamp_min(x2 - x1, 0)) * (torch.clamp_min(y2 - y1, 0))    # 计算交集面积
    union_area = torch.sum(area) - inter_area                             # 计算并集面积
    iou = inter_area / (union_area + 1e-6)                              # 计算交并比

    return iou


def generalized_AABB_iou(box_1, box_2):
    # 分别是第一个矩形左右上下的坐标
    x1, x2, y1, y2 = box_1
    x3, x4, y3, y4 = box_2
    iou = AABB_iou(box_1, box_2)
    area_bound = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
    area_1 = (x2 - x1) * (y1 - y2)
    area_2 = (x4 - x3) * (y3 - y4)
    sum_area = area_1 + area_2

    w1 = x2 - x1   # 第一个矩形的宽
    w2 = x4 - x3   # 第二个矩形的宽
    h1 = y1 - y2
    h2 = y3 - y4
    w = min(x1, x2, x3, x4) + w1 + w2 - max(x1, x2, x3, x4)     # 交集的宽
    h = min(y1, y2, y3, y4) + h1 + h2 - max(y1, y2, y3, y4)     # 交集的高
    inter_area = w * h                                          # 交集的面积
    union_area = sum_area - inter_area                          # 并集的面积

    end_area = (area_bound - union_area) / area_bound    # 闭包区域中不属于两个框的区域占闭包区域的比重
    giou = iou - end_area
    return giou


def distance_AABB_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    distance_iou = torch.zeros((rows, cols))
    if rows * cols == 0:  #
        return distance_iou
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        distance_iou = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    distance_iou = inter_area / union - (inter_diag) / outer_diag
    distance_iou = torch.clamp(distance_iou, min=-1.0, max=1.0)
    if exchange:
        distance_iou = distance_iou.T
    return distance_iou


# --------------------------- RotatedRect ---------------------------

def _check_corners(box_1, box_2):
    """
    计算两个旋转矩形的距离IOU
    Args:
        box_1 (torch.Tensor): 可以是 rotated_rect 也可以是 corners
        box_2 (torch.Tensor):
    Returns:
    """
    if box_1.dim() == 1 and box_1.shape == torch.Size([5]):
        corners_1 = rotated_rect_to_corners(box_1)
        corners_2 = rotated_rect_to_corners(box_2)
    elif box_1.dim() == 2 and box_1.shape[-2:] == torch.Size([4, 2]):
        corners_1 = box_1
        corners_2 = box_2
    else:
        raise ValueError(f"rotated_rect shape should be torch.Size([5]) or torch.Size([4, 2]),"
                         f" instead of {box_1.shape} and {box_2.shape}")

    return corners_1, corners_2


def _check_corners_batch(rotated_rects_ref, rotated_rects_obj):
    """
    将旋转矩形转换为角点
    Args:
        rotated_rects_ref:
        rotated_rects_obj:
    Returns:

    """
    # [5] -> [1, 5]  |  [B, 5]
    if rotated_rects_ref.dim() == 1:
        rotated_rects_ref = rotated_rects_ref.unsqueeze(0)
    if rotated_rects_obj.dim() == 1:
        rotated_rects_obj = rotated_rects_obj.unsqueeze(0)

    # [B, 5] -> [B, 4, 2]
    corners_ref = rotated_rect_to_corners(rotated_rects_ref)
    corners_obj = rotated_rect_to_corners(rotated_rects_obj)

    return corners_ref, corners_obj


def rotated_rect_to_corners(rotated_rect):
    """
    包围盒转化为角点
    Args:
        rotated_rect (torch.Tensor | list(torch.Tensor)): torch.tensor([x, y, w, h, angle])，其中 angle 为弧度制

    Returns:
        顺时针方向返回角点位置
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    if isinstance(rotated_rect, list):
        rotated_rect = torch.stack(rotated_rect)

    if rotated_rect.dim() == 1:
        rotated_rect = rotated_rect.unsqueeze(0)

    if rotated_rect.dim() != 1 and rotated_rect.dim() != 2:
        raise ValueError("rotated_rect should be a tensor with 1 or 2 dimensions, "
                         f"instead of dims={rotated_rect.dim()}, shape={rotated_rect.shape}")

    num_rect = rotated_rect.shape[0]
    center, size, angle = rotated_rect.split((2, 2, 1), dim=-1)
    center = center.unsqueeze(-2)                               # [num_rect, 2] -> [num_rect, 1, 2]
    size = size.unsqueeze(-2)                                   # [num_rect, 2] -> [num_rect, 1, 2]

    half_size = size / 2
    a_cos, a_sin = torch.cos(angle), torch.sin(angle)           # [num_rect, 1]
    k = (torch
         .stack(tensors=[a_cos, -a_sin, a_sin, a_cos], dim=-1)  # [num_rect, 1, 4]
         .view(num_rect, 2, 2))                                 # [num_rect, 2, 2]
    offset = \
        (torch
         .tensor([[-1, -1, 1, 1],  # width
                  [-1, 1, 1, -1]   # height
                  ])                                            # [1, 4, 2]
         .repeat(num_rect, 1, 1))                               # [num_rect, 4, 2]

    corners = center + (half_size * offset.transpose(-2, -1)) @ k.transpose(-2, -1)

    return corners
    # 当 rotated_rect.dim == 1 时，代码可以转换为：
    #
    # center, size, angle = rotated_rect.split((2, 2, 1), dim=-1)
    # half_size = size / 2
    # a_cos, a_sin = math.cos(angle), math.sin(angle)
    # k = torch.tensor([[a_cos, a_sin], [-a_sin, a_cos]])
    # offset = torch.tensor([[-1, -1, 1, 1],  # width
    #                        [-1, 1, 1, -1]   # height
    #                        ])
    #
    # corners = center + (half_size * offset.T) @ k.T
    # return corners


def is_point_inside_rotated_rect(point, corners):
    """
    判断点在四边形（矩形）内
    Args:
        point (torch.Tensor):
        corners (torch.Tensor):

    Returns:

    """
    AB = corners[1] - corners[0]
    AD = corners[3] - corners[0]
    AP = point - corners[0]

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    AB_dot_AB = dot(AB, AB)
    AB_dot_AP = dot(AB, AP)
    AD_dot_AD = dot(AD, AD)
    AD_dot_AP = dot(AD, AP)

    return AB_dot_AB >= AB_dot_AP >= 0 and AD_dot_AD >= AD_dot_AP >= 0

    # if corners.dim() == 2:
    #     corners = corners.unsqueeze(0)
    #
    # if corners.dim() != 2 and corners.dim() != 3:
    #     raise ValueError("corners should be a tensor with 2 or 3 dimensions")
    #
    # AB = corners[:, 1] - corners[:, 0]
    # AD = corners[:, 3] - corners[:, 0]
    # AP = points - corners[:, 0]
    #
    # def dot(a, b):
    #     return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    #
    # AB_dot_AB = dot(AB, AB)
    # AB_dot_AP = dot(AB, AP)
    # AD_dot_AD = dot(AD, AD)
    # AD_dot_AP = dot(AD, AP)
    #
    # zero = torch.zeros_like(AB_dot_AB)
    # all_conditions = torch.stack(
    #     [
    #         AB_dot_AB >= AB_dot_AP,
    #         AB_dot_AP >= zero,
    #         AD_dot_AD >= AD_dot_AP,
    #         AD_dot_AP >= zero
    #     ])
    # return torch.all(all_conditions, dim=0)


def line_segment_intersection(line_segment_1, line_segment_2):
    """
    计算两条线段的交点
    Args:

    Returns:

    """
    A, B = line_segment_1
    C, D = line_segment_2

    AB = B - A
    AC = C - A
    AD = D - A
    BD = D - B
    BC = C - B

    def _cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    acd = _cross(AC, AD) > 0
    bcd = _cross(BC, BD) > 0
    if acd != bcd:
        abc = _cross(AB, AC) > 0
        abd = _cross(AB, AD) > 0
        if abc != abd:
            DC = D - C
            cross_AB = _cross(A, B)
            cross_CD = _cross(C, D)
            DH = _cross(DC, AB)
            intersection_point = (cross_AB * DC - AB * cross_CD) / DH
            return True, intersection_point
    return False, torch.tensor([0, 0])


def sort_vertex_in_convex_polygon(points):
    """
    凸多边形顶点排序
    """
    if len(points) <= 0:
        return

    center = torch.mean(points, dim=0)

    points -= center
    points = points[torch.sort(points.T[1], dim=0).indices]
    points = points[torch.sort(points.T[0], dim=0).indices]
    points += center

    return points


def get_area(points):
    """
    求凸多边形面积，将多边形转化为多个三角形面积之和
    """
    def _cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def _triangle_area(a, b, c):
        return abs(_cross(a - c, b - c) / 2)

    area_val = 0.0
    for i in range(len(points) - 2):
        area_val += _triangle_area(points[0], points[i + 1], points[i + 2])
    return area_val


def get_max_distance(corners_1, corners_2):
    max_distance = 0
    for i in range(4):
        for j in range(4):
            distance = torch.norm(corners_1[i] - corners_2[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def get_center_distance(corners_1, corners_2):
    center_1 = torch.mean(corners_1, dim=0)
    center_2 = torch.mean(corners_2, dim=0)
    return torch.norm(center_1 - center_2)


def OBB_iou(box_1, box_2):
    """
    计算两批旋转矩形的IOU
    Args:
        box_1 (torch.Tensor): 可以是 rotated_rect 也可以是 corners
        box_2 (torch.Tensor):
    Returns:

    """
    corners_1, corners_2 = _check_corners(box_1, box_2)
    area = get_area(corners_1) + get_area(corners_2)

    points = []
    for corner_1 in corners_1:
        if is_point_inside_rotated_rect(corner_1, corners_2):
            points.append(corner_1)
    for corner_2 in corners_2:
        if is_point_inside_rotated_rect(corner_2, corners_1):
            points.append(corner_2)
    for i in range(4):
        for j in range(4):
            any_inside, point = line_segment_intersection(
                corners_1[[i, (i + 1) % 4]],
                corners_2[[j, (j + 1) % 4]])
            if any_inside:
                points.append(point)

    if len(points) == 0:
        return 0.0

    points = torch.stack(points)
    points = sort_vertex_in_convex_polygon(points)

    inter_area = get_area(points)
    union_area = area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou


def OBB_iou_batch(rotated_rects_ref, rotated_rects_obj):
    corners_ref, corners_obj = _check_corners_batch(rotated_rects_ref, rotated_rects_obj)

    ref_batch_size = rotated_rects_ref.shape[0]
    obj_batch_size = rotated_rects_obj.shape[0]

    ious = []
    for i_batch in range(ref_batch_size):
        for j_batch in range(obj_batch_size):
            iou = OBB_iou(corners_ref[i_batch], corners_obj[j_batch])
            ious.append(iou)

    return torch.tensor(ious).view(ref_batch_size, obj_batch_size)


def distance_OBB_iou(box_1, box_2):
    """
    计算两个旋转矩形的距离IOU
    Args:
        box_1 (torch.Tensor): 可以是 rotated_rect 也可以是 corners
        box_2 (torch.Tensor):
    Returns:
    """
    corners_1, corners_2 = _check_corners(box_1, box_2)

    iou = OBB_iou(box_1, box_2)
    d = get_max_distance(corners_1, corners_2)
    c = get_center_distance(corners_1, corners_2)
    return iou - (c / (d + 1e-6)) ** 2


def distance_OBB_iou_batch(rotated_rects_ref, rotated_rects_obj):
    corners_ref, corners_obj = _check_corners_batch(rotated_rects_ref, rotated_rects_obj)

    ref_batch_size = rotated_rects_ref.shape[0]
    obj_batch_size = rotated_rects_obj.shape[0]

    ious = []
    for i_batch in range(ref_batch_size):
        for j_batch in range(obj_batch_size):
            iou = distance_OBB_iou(corners_ref[i_batch], corners_obj[j_batch])
            ious.append(iou)

    return torch.tensor(ious).view(ref_batch_size, obj_batch_size)


def obbox_iou(pred_boxes, target_boxes, iou_type='DIoU'):
    if iou_type == 'IoU':
        return OBB_iou_batch(pred_boxes, target_boxes)
    elif iou_type == 'DIoU':
        return distance_OBB_iou_batch(pred_boxes, target_boxes)
    else:
        raise ValueError('unknown iou type')



if __name__ == '__main__':
    # 计算两个旋转矩形的IOU
    rotated_rect_0 = torch.tensor([5, 5, 5, 5, 0])
    rotated_rect_1 = torch.tensor([5, 5, 10, 10, 0])
    rotated_rect_2 = torch.tensor([0, 0, 10, 10, 60 * math.pi / 180])
    rotated_rect_3 = torch.tensor([10, 10, 10, 10, 45 * math.pi / 180])
    rotated_rect_4 = torch.tensor([10, 10, 10, 10, 15 * math.pi / 180])
    # points = rotated_rect_to_corners(torch.stack([rotated_rect_1, rotated_rect_2, rotated_rect_3]))
    print(distance_OBB_iou_batch(
        torch.stack([rotated_rect_0, rotated_rect_1]),
        torch.stack([rotated_rect_2, rotated_rect_3, rotated_rect_4])))
