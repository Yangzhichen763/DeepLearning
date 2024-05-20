import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


class AABBboxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores, iou_type="DIoU"):
        """
        计算 AABB 边界框的交并比损失值
        Args:
            pred_bboxes: 预测的边界框。可以取 shape=[B, N, 4]，或者 shape=[..., 4]
            target_bboxes: 真实的边界框。可以取 shape=[B, x, 4]，或者 shape=[..., 4]
            target_scores: 真实的边界框的置信度。可以取 shape=[B, N, x] | [B, N] | [B] | []（标量）
            iou_type:
        Returns:
            loss: 交并比损失值，shape=[B, N]
            iou: 边界框的交并比，shape=[B, N, x]
        """
        weight_shape = (*pred_bboxes.shape[:-2], target_bboxes.shape[-2])   # (B, N, x)
        weight, weight_sum = self.check_target_scores(weight_shape, target_scores, pred_bboxes.device)

        ious = []
        for i in range(target_bboxes.shape[-2]):
            iou = aabbox_iou(pred_bboxes,
                             target_bboxes[..., i, :].unsqueeze(dim=-2),
                             iou_type=iou_type)                              # [B, N, 1]
            ious.append(iou)
        ious = torch.cat(ious, dim=-1)                                      # [B, N, x]
        loss_iou = ((1.0 - ious) * weight).sum(dim=-1) / weight_sum         # [B, N]

        return loss_iou, ious

    @staticmethod
    def check_target_scores(weight_shape, target_scores, device):
        weight = None
        weight_sum = 1.0
        if target_scores is None:
            target_scores = torch.ones(weight_shape, device=device)
            weight_sum = weight_shape[-1]
        elif isinstance(target_scores, float) or isinstance(target_scores, int):
            weight = torch.ones(weight_shape, device=device) * target_scores
            weight_sum = target_scores * weight_shape[-1]
        elif isinstance(target_scores, torch.Tensor):
            # 转换成 [B, N, x] 形式
            if target_scores.dim() == 0:
                weight = target_scores
            elif target_scores.dim() == 1 and target_scores.shape[0] == weight_shape[0]:
                weight = target_scores.view(-1, 1, 1).repeat(1, weight_shape[-2], weight_shape[-1])
            elif target_scores.dim() == 2 and target_scores.shape[0] == weight_shape[0]:
                weight = target_scores.unsqueeze(dim=-1).repeat(1, 1, weight_shape[-1])
            else:
                raise ValueError("target_scores a tensor must be with shape=[B, N, 1], shape=[B, N, x] or shape=[], "
                                 f"instead of {target_scores.shape} with content {target_scores}")
            print(weight.sum(dim=-1).shape, weight_shape[-1])
            weight_sum = torch.clamp(
                weight.sum(dim=-1),
                min=torch.tensor(-weight_shape[-1], device=device),
                max=torch.tensor(weight_shape[-1], device=device))
            print(weight_sum.shape)
        if weight is None:
            raise ValueError("target_scores must be a tensor or a float, "
                             f"instead of {target_scores} with type {type(target_scores)}")

        return weight, weight_sum

    @staticmethod
    def check_image_size(image_size, device):
        """
        将 image_size 转换成二维 torch.Tensor，即类似 (256, 256) 形式的数据
        Args:
            image_size:
            device:
        Returns:
        """
        if isinstance(image_size, int):
            image_size = torch.tensor((image_size, image_size), device=device)
        elif isinstance(image_size, (list, tuple)):
            image_size = torch.tensor(image_size, device=device)
        elif isinstance(image_size, torch.Tensor):
            image_size = image_size.detach().to(device)
        else:
            raise ValueError("image_size must be a list, tuple or tensor")

        if image_size.shape[-1] >= 2:
            image_size = image_size[-2:]
        if image_size.dim() > 1 or image_size.shape[-1] != 2:
            raise ValueError(f"image_size must be a tensor with shape=[2], "
                             f"instead of {image_size.shape} with content {image_size}")

        return image_size


class OBBLoss(AABBboxLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores=None, iou_type="IoU"):
        """
        计算 AABB 边界框的交并比损失值
        Args:
            pred_bboxes: 预测的边界框。可以取 shape=[B, N, 5]，或者 shape=[..., 5]
            target_bboxes: 真实的边界框。可以取 shape=[B, x, 5]，或者 shape=[..., 5]
            target_scores: 真实的边界框的置信度。可以取 shape=[B, N, x], shape=[B, N, 1]，可以是标量，但是没有意义
            iou_type:
        Returns:
            loss: 交并比损失值，shape=[B, N]
            iou: 边界框的交并比，shape=[B, N, x]
        """
        weight_shape = (*pred_bboxes.shape[:-1], target_bboxes.shape[-2])   # (B, N, x)
        weight, weight_sum = self.check_target_scores(weight_shape, target_scores, pred_bboxes.device)

        ious = []
        for i in range(target_bboxes.shape[-2]):
            iou = obbox_iou(pred_bboxes,
                            target_bboxes[..., i, :].unsqueeze(dim=-2),
                            iou_type=iou_type)                              # [B, N, 1]
            ious.append(iou)
        ious = torch.cat(ious, dim=-1)                                      # [B, N, x]
        loss_iou = ((1.0 - ious) * weight).sum(dim=-1) / weight_sum         # [B, N]

        return loss_iou, ious


class OBBLoss_without_position(OBBLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores=None, iou_type="IoU"):
        """
        计算 AABB 边界框的交并比损失值
        Args:
            pred_bboxes: 预测的边界框。可以取 shape=[B, N, 5]，或者 shape=[..., 5]
            target_bboxes: 真实的边界框。可以取 shape=[B, x, 5]，或者 shape=[..., 5]
            target_scores: 真实的边界框的置信度。可以取 shape=[B, N, x], shape=[B, N, 1]，可以是标量，但是没有意义
            iou_type:
        Returns:
            loss: 交并比损失值，shape=[B, N]
            iou: 边界框的交并比，shape=[B, N, x]
        """
        pred_bboxes, target_bboxes = self.exclude_position(pred_bboxes), self.exclude_position(target_bboxes)

        return super().forward(pred_bboxes, target_bboxes, target_scores, iou_type)

    def exclude_position(self, rr):
        center, size, angle = rr.split(split_size=(2, 2, 1), dim=-1)
        zeros = torch.zeros_like(center)
        rr = torch.cat([zeros, size, angle], dim=-1)
        return rr


class EllipseLoss(OBBLoss):
    def __init__(self, image_size):
        super(EllipseLoss, self).__init__()
        self.pred_loss_func = OBBLoss()
        # self.rotation_loss_func = OBBLoss_without_position()
        self.score_loss_func = nn.CrossEntropyLoss()
        self.image_size = image_size

    def __call__(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测的边界框。可以取 shape=[batch_size, 64, 5]
            target (torch.Tensor): 目标的边界框。可以取 shape=[batch_size, x, 5]
        Returns:
            l_iou: 所有 batch 的交并比损失值的平均值，为标量
            iou: 每个 batch 边界框的交并比，shape=[B, N, x]
            pred: 每个 batch 预测的边界框，shape=[B, k, 5] 其中 k = x
        """
        # 0,1 是椭圆中心坐标
        # 2,3 是椭圆长短轴长度
        # 4   是椭圆旋转角度
        device = pred.device
        batch_size = pred.shape[0]      # [B, N, 5] 中的 B
        dim_rr = pred.shape[-1]         # [B, N, 5] 中的 5
        num_target = target.shape[-2]   # [B, x, 5] 中的 x

        # 计算交并比损失
        pred = self.regress_rr(pred, self.image_size)      # box 归一化处理，使得 box 在图像范围内（去除负值等）
        ls_iou, ious = self.pred_loss_func(pred, target, 1)                 # [B, N], [B, N, x]
        # ls_rotation, rots = self.rotation_loss_func(pred, target, 1)        # [B, N]
        ious_max, _ = ious.max(dim=-1)                                      # [B, N, x] -> [B, N] 找到 x 中的最大值
        ious_max = ious_max.unsqueeze(dim=-1)                               # [B, N] -> [B, N, 1]
        pred, pred_ious = self.get_argmax_pred(pred, ious_max, target)      # [B, k, 5], [B, k, 1] 其中 k = x

        # 计算 score 损失，以下损失都是标量
        l_iou = ious.mean()
        # l_rotation = rots.mean()
        pred_ious = pred_ious.squeeze(dim=-1).transpose(0, 1)
        l_score = self.score_loss_func(pred_ious, torch.ones_like(pred_ious))
        if l_score.isnan().any():
            print("l_score is nan")
            exit()
        loss = l_iou + l_score / batch_size

        return loss, ious, pred               # 标量, [B, N, x], [B, k, 5] 其中 k = x

    @staticmethod
    def split_pred(pred):
        """
        将 pred 拆分为 rotated_rect 和 score
        Args:
            pred: 预测得到的边界框，shape=[batch_size, num_preds, 6]
        Returns:
            pred_rotated_rects: 预测得到的边界框，shape=[batch_size, num_preds, 5]
            pred_scores: 预测得到的分数，shape=[batch_size, num_preds, 1]
        """
        pred_rotated_rects, pred_scores = pred.split(split_size=(5, 1), dim=-1)
        return pred_rotated_rects, pred_scores

    @staticmethod
    def get_argmax_pred(pred_rr, pred_scores, target_rr):
        """
        [B, N, 5] -> [B, k, 5]，其中 k 对应着 target 中边框的数量
        Args:
            pred_rr: 预测得到的边界框，shape=[B, N, 5]
            pred_scores: 预测得到的分数，shape=[B, N, 1]
            target_rr: 真实的边界框，shape=[B, x, 5]
        Returns:
            pred_rr: 分数前 k 高的预测边界框
            pred_scores: 分数前 k 高的预测分数
            其中 k 对应着 target 中边框的数量
        """
        num_targets, num_preds = target_rr.shape[-2], pred_rr.shape[-2]

        # 获取 topk 最高分的预测边界框
        if num_targets < num_preds:
            pred_scores, top_indices = pred_scores.topk(k=num_targets, dim=-2)
            pred_rr = torch.stack(
                [
                    pred_rr[..., i, index.squeeze(dim=-1), :]
                    for i, index in enumerate(top_indices)
                ])
        # print("pred:", pred.shape, "target:",  target.shape, "topk:", pred_rr.shape)

        return pred_rr, pred_scores

    @staticmethod
    def regress_rr(rotated_rect, image_size):
        image_size = AABBboxLoss.check_image_size(image_size, rotated_rect.device)

        # 回归到原图坐标系
        center, size, angle = rotated_rect.split(split_size=(2, 2, 1), dim=-1)

        factor = 11.1

        def _normalize(x):
            return (x * factor / image_size).sigmoid()

        center = _normalize(center) * image_size
        size = (_normalize(size) ** 2 * image_size).clamp_min(1.0)
        angle = torch.sin(angle * math.pi / 180).asin() * 180 / math.pi
        rotated_rect = torch.cat([center, size, angle], dim=-1)
        return rotated_rect

    @staticmethod
    def regress_pred_scores(pred_scores):
        scores = pred_scores.sigmoid()
        return scores


# --------- 以下代码来自 https://github.com/ultralytics/ultralytics ----------
def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def aabbox_iou(box1, box2, xywh=True, iou_type="IoU", eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if iou_type in ["CIou", "DIoU", "GIoU"]:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if iou_type in ["CIou", "DIoU"]:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if iou_type in ["CIou"]:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def obbox_iou(obb1, obb2, iou_type="IoU", eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if iou_type in ["CIoU"]:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[..., 2:4].pow(2) / 12, boxes[..., 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
# --------- 以上代码来自 https://github.com/ultralytics/ultralytics ----------


# --------------------------- AABB ---------------------------

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


# --------------------------- OBB ---------------------------

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
    elif rotated_rect.dim() == 3 and rotated_rect.shape[-1] == 5:
        rotated_rect = rotated_rect.flatten(start_dim=0, end_dim=-2)

    if rotated_rect.dim() not in [1, 2, 3]:
        raise ValueError("rotated_rect should be a tensor with 1, 2 or 3 dimensions, "
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


def _obbox_iou(pred_boxes, target_boxes, iou_type='DIoU'):
    if iou_type == 'IoU':
        return OBB_iou_batch(pred_boxes, target_boxes)
    elif iou_type == 'DIoU':
        return distance_OBB_iou_batch(pred_boxes, target_boxes)
    else:
        raise ValueError('unknown iou type')



if __name__ == '__main__':
    batch_size = 8
    _pred = torch.randn(batch_size, 64, 5) * 8
    _target = EllipseLoss.regress_rr(torch.randn(batch_size, 1, 5), (256, 256))
    loss, iou, _ = EllipseLoss((256, 256))(_pred, _target)
    print(loss)
    exit()
    # 计算两个旋转矩形的IOU
    rotated_rect_with_scores_0 = torch.tensor([5, 5, 5, 5, 0, 1.2])
    rotated_rect_with_scores_1 = torch.tensor([5, 5, 10, 10, 0, 1.2])
    rotated_rect_with_scores_2 = torch.tensor([0, 0, 10, 10, 60 * math.pi / 180, 1.2])
    rotated_rect_with_scores_3 = torch.tensor([10, 10, 10, 10, 45 * math.pi / 180, 1.2])
    rotated_rect_with_scores_4 = torch.tensor([10, 10, 10, 10, 15 * math.pi / 180, 1.2])
    rotated_rect_with_scores_5 = torch.tensor([10, 10, 10, 10, 30 * math.pi / 180, 1.2])

    rotated_rect_0 = torch.tensor([5, 5, 5, 5, 0])
    rotated_rect_1 = torch.tensor([5, 5, 10, 10, 0])
    rotated_rect_2 = torch.tensor([0, 0, 10, 10, 60 * math.pi / 180])
    rotated_rect_3 = torch.tensor([10, 10, 10, 10, 45 * math.pi / 180])
    rotated_rect_4 = torch.tensor([10, 10, 10, 10, 15 * math.pi / 180])
    rotated_rect_5 = torch.tensor([10, 10, 10, 10, 30 * math.pi / 180])

    rotated_rect_pred = torch.stack([
        rotated_rect_with_scores_0,
        rotated_rect_with_scores_1,
        rotated_rect_with_scores_2,
        rotated_rect_with_scores_3,
        rotated_rect_with_scores_4,
        rotated_rect_with_scores_5]).repeat(4, 1, 1)
    rotated_rect_target = torch.stack([
        rotated_rect_0,
        rotated_rect_1,
        rotated_rect_2,
        rotated_rect_3,
        rotated_rect_4,
        rotated_rect_5]).repeat(4, 1, 1)
    print(rotated_rect_pred.shape, rotated_rect_target.shape)
    _loss_func = EllipseLoss()
    _losses, _ious = _loss_func(rotated_rect_pred, rotated_rect_target)
    print(_losses)
