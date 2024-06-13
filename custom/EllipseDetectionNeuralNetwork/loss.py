import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lossFunc.FunctionalLoss
from lossFunc import LogLoss, WHRatioLoss
from loss_part import *

import math
from lossFunc.BBLoss import *


class AABBboxLoss(nn.Module):
    def __init__(self):
        super(AABBboxLoss, self).__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores=1, iou_type="CIoU"):
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
        weight_shape = (*pred_bboxes.shape[:-2], target_bboxes.shape[-2])  # (B, N, x)
        weight, weight_sum = self.check_target_scores(weight_shape, target_scores, pred_bboxes.device)

        ious = []
        for i in range(target_bboxes.shape[-2]):
            iou = aabbox_iou(pred_bboxes,
                             target_bboxes[..., i, :].unsqueeze(dim=-2),
                             iou_type=iou_type)  # [B, N, 1]
            ious.append(iou)
        ious = torch.cat(ious, dim=-1)  # [B, N, x]
        loss_iou = ((1.0 - ious) * weight).sum(dim=-1) / weight_sum  # [B, N]

        return loss_iou, ious

    @staticmethod
    def check_target_scores(weight_shape, target_scores, device):
        """
        将得分值转换为权重值
        Args:
            weight_shape:
            target_scores:
            device:

        Returns:

        """
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
        super(OBBLoss, self).__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores=1, iou_type="CIoU"):
        """
        计算 OBB 边界框的交并比损失值
        Args:
            pred_bboxes: 预测的边界框。可以取 shape=[B, N, 5]，或者 shape=[..., 5]
            target_bboxes: 真实的边界框。可以取 shape=[B, x, 5]，或者 shape=[..., 5]
            target_scores: 真实的边界框的置信度。可以取 shape=[B, N, x], shape=[B, N, 1]，可以是标量，但是没有意义
            iou_type:
        Returns:
            loss: 交并比损失值，shape=[B, N]
            iou: 边界框的交并比，shape=[B, N, x]，交并比越大，说明预测的边界框与真实的边界框越相似
            其中 loss.mean() = 1 - iou
        """
        weight_shape = (*pred_bboxes.shape[:-1], target_bboxes.shape[-2])  # (B, N, x)
        weight, weight_sum = self.check_target_scores(weight_shape, target_scores, pred_bboxes.device)

        ious = []
        for i in range(target_bboxes.shape[-2]):
            iou = obbox_iou(pred_bboxes,
                            target_bboxes[..., i, :].unsqueeze(dim=-2),
                            iou_type=iou_type)  # [B, N, 1]
            ious.append(iou)
        ious = torch.cat(ious, dim=-1)  # [B, N, x]
        loss_iou = ((1.0 - ious) * weight).sum(dim=-1) / weight_sum  # [B, N]

        return loss_iou, ious


class OBBConcentricLoss(OBBLoss):
    def __init__(self):
        super(OBBConcentricLoss, self).__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores=1, iou_type="CIoU"):
        """
        计算 OBB 边界框的交并比损失值，
        如果两个 OBB 不是同心框，则计算损失时强制归到同心
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


class OBBDirectionLoss(nn.Module):
    def __init__(self):
        super(OBBDirectionLoss, self).__init__()
        self.loss_func = WHRatioLoss()

    def forward(self, pred, target):
        """
        计算 OBB 边界框的方向损失值，方向一致 loss -> 0，方向不一致 loss -> 1
        Args:
            pred (torch.Tensor): 长宽合法（已回归过的）的预测边界框。可以取 shape=[batch_size, 64, 5]
            target (torch.Tensor): 长宽合法（已回归过的）的目标边界框。可以取 shape=[batch_size, x, 5]

        Returns:

        """
        w_p, h_p, a_p = pred[..., 2:].split(split_size=1, dim=-1)
        w_t, h_t, a_t = target[..., 2:].split(split_size=1, dim=-1)
        loss, score = self.loss_func(w_p / h_p, w_t / h_t, a_p, a_t)
        return loss, score


class EllipseLoss(OBBLoss):
    def __init__(self, image_size):
        super(EllipseLoss, self).__init__()
        self.pred_loss_func = OBBLoss()
        self.size_loss_func = OBBConcentricLoss()
        self.rotation_loss_func = OBBDirectionLoss()
        self.score_loss_func = LogLoss()
        self.image_size = image_size

    def __call__(self, pred, target, regress_pred=True):
        """
        Args:
            pred (torch.Tensor): 预测的边界框。可以取 shape=[batch_size, 64, 5]
            target (torch.Tensor): 合法的目标的边界框。可以取 shape=[batch_size, x, 5]
            regress_pred (bool): 是否对预测的边界框进行回归（合法化），默认 True
        Returns:
            l_iou: 所有 batch 的交并比损失值的平均值，为标量
            l_scores_rotation: 所有 batch 的方向损失值的平均值，为标量
            iou: 每个 batch 边界框的交并比，shape=[B, k, 1]
            pred: 每个 batch 预测的边界框，shape=[B, k, 5] 其中 k = x
        """
        # 0,1 是椭圆中心坐标
        # 2,3 是椭圆长短轴长度
        # 4   是椭圆旋转角度
        device = pred.device
        batch_size = pred.shape[0]  # [B, N, 5] 中的 B
        dim_rr = pred.shape[-1]  # [B, N, 5] 中的 5
        num_target = target.shape[-2]  # [B, x, 5] 中的 x

        # 计算交并比损失
        if regress_pred:
            pred = self.regress_rr(pred, self.image_size)  # box 归一化处理，使得 box 在图像范围内（去除负值等）
        _, ious = self.pred_loss_func(pred, target)  # [B, N], [B, N, x] = ~0, ~1

        ious_max, _ = ious.max(dim=-1)  # [B, N, x] -> [B, N] 找到 x 中的最大值
        ious_max = ious_max.unsqueeze(dim=-1)  # [B, N] -> [B, N, 1]
        pred, pred_ious = self.get_argmax_pred(pred, ious_max, target)  # [B, k, 5], [B, k, 1] 其中 k = x

        loss_size, pred_size = self.size_loss_func(pred, target)  # [B, N], [B, N, x] = ~0, ~1
        loss_rots, pred_rots = self.rotation_loss_func(pred, target)  # [B, N], [B, N, x] = ~0, ~0

        # 计算 score 损失，以下损失都是标量
        def _iou_to_loss(_ious):
            scores = _ious.squeeze(dim=-1).transpose(0, 1)
            l_score = self.score_loss_func(scores, torch.ones_like(scores))
            return l_score

        l_scores_iou = 1 - pred_ious.mean()  # _iou_to_loss(pred_ious)
        l_scores_size = 1 - pred_size.mean()  # _iou_to_loss(pred_size)
        l_scores_rotation = loss_rots.mean()

        l_scores_iou *= 1
        l_scores_size *= 2
        l_scores_rotation = lossFunc.FunctionalLoss.LogLoss()(l_scores_rotation) * 12
        print(l_scores_iou, l_scores_size, l_scores_rotation)
        loss = l_scores_iou + l_scores_size + l_scores_rotation  # + l_scores_corner

        return loss, l_scores_rotation, pred_ious, pred  # 标量, 标量, [B, k, 1], [B, k, 5] 其中 k = x

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
        if num_targets <= num_preds:
            pred_scores, top_indices = pred_scores.topk(k=num_targets, dim=-2)  # [B, k, 1], [B, k, 1]

            top_indices = (top_indices  # [B, k, 1]
                           .repeat_interleave(repeats=pred_rr.shape[-1], dim=-1))  # [B, k, 5]
            # 得到 top_indices 索引值对应的每个边框的 rr 值
            pred_rr = torch.gather(pred_rr, dim=-2, index=top_indices)  # [B, k, 5] -> [B, k, 5]
            # 修改前的方案：
            # pred_rr = torch.stack(
            #     [
            #         pred_rr[..., i, index.squeeze(dim=-1), :]
            #         for i, index in enumerate(top_indices)
            #     ])

        return pred_rr, pred_scores

    @staticmethod
    def regress_rr(rotated_rect, image_size):
        image_size = AABBboxLoss.check_image_size(image_size, rotated_rect.device)

        # 回归到原图坐标系
        center, size, angle = rotated_rect.split(split_size=(2, 2, 1), dim=-1)

        # std_center, std_size, std_angle = torch.std(center), torch.std(size), torch.std(angle)
        # print(std_center, std_size, std_angle)
        factor = 11.1

        def _normalize(x):
            return (x * factor / image_size).sigmoid()

        def _ease_in_out_cubic(x):
            return 3 * x ** 2 - 2 * x ** 3

        center = _normalize(center) * image_size
        size = (_normalize(size) * image_size).clamp(torch.ones(1).to(rotated_rect.device),
                                                     image_size.max())  # (_normalize(size) ** 2 * image_size)clamp(1.0, image_size)
        # angle = torch.sin(angle).asin()
        rotated_rect = torch.cat([center, size, angle], dim=-1)
        return rotated_rect

    @staticmethod
    def regress_pred_scores(pred_scores):
        scores = pred_scores.sigmoid()
        return scores


if __name__ == '__main__':
    _batch_size = 8
    _pred = torch.tensor([[3, 13, 28.3614, 58.2907, 1.084917]]).view(1, -1, 5).repeat(_batch_size, 1, 1)
    _target = torch.tensor([2.2, 13, 31.6460047497, 33.5405781611, 1.060262]).view(1, -1, 5).repeat(_batch_size, 1, 1)
    loss, loss_rotation, iou, _ = EllipseLoss((256, 256))(_pred, _target, regress_pred=False)
    print(loss, loss_rotation)
    exit()

    batch_size = 8
    _position = torch.randn(batch_size, 1, 2) * 1
    _size = torch.randn(batch_size, 1, 2) * 16
    _angle = torch.randn(batch_size, 1, 1) * 2 * math.pi
    _pred = torch.cat([_position, _size, _angle], dim=-1)
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
