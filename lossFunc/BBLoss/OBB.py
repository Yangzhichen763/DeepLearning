"""
OBB (Oriented Bounding Box)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def obb_to_corners(rotated_rect):
    """
    包围盒转化为角点，即 xywhr -> xy sigma
    Args:
        rotated_rect (torch.Tensor): torch.tensor([x, y, w, h, angle])，其中 angle 为弧度制

    Returns:
        (torch.tensor): torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        顺时针方向返回角点位置
        [x, y, w, h, angle] -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        [B, N, 5] -> [B, N, 4, 2]
    """
    if rotated_rect.dim() == 1:
        rotated_rect = rotated_rect.unsqueeze(0)
    # elif rotated_rect.dim() == 3 and rotated_rect.shape[-1] == 5:
    #     rotated_rect = rotated_rect.flatten(start_dim=0, end_dim=-2)

    # if rotated_rect.dim() not in [1, 2, 3]:
    #     raise ValueError("rotated_rect should be a tensor with 1, 2 or 3 dimensions, "
    #                      f"instead of dims={rotated_rect.dim()}, shape={rotated_rect.shape}")

    num_rect = rotated_rect.shape[0]
    center, size, angle = rotated_rect[..., 0:2], rotated_rect[..., 2:4], rotated_rect[..., 4:]
    center = center.unsqueeze(-2)                               # [num_rect, 2] -> [num_rect, 1, 2]
    size = size.unsqueeze(-2)                                   # [num_rect, 2] -> [num_rect, 1, 2]

    half_size = size / 2
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)           # [num_rect, 1, 1]
    k = (torch
         .cat(tensors=[cos_a, -sin_a, sin_a, cos_a], dim=-1)    # [num_rect, 1, 4]
         .view(-1, 2, 2))                                       # [num_rect, 2, 2]
    offset = \
        (torch
         .tensor(
            data=[[-1, -1, 1, 1],   # width
                  [-1, 1, 1, -1]],  # height
            device=rotated_rect.device)                         # [1, 2, 4]
         .repeat(num_rect, 1, 1))                               # [num_rect, 2, 4]

    corners = center + (half_size * offset.transpose(-2, -1)) @ k.transpose(-2, -1)

    return corners
    # 当 rotated_rect.dim == 1 时，代码可以转换为：
    #
    # center, size, angle = rotated_rect.split((2, 2, 1), dim=-1)
    # half_size = size / 2
    # cos_a, sin_a = math.cos(angle), math.sin(angle)
    # k = torch.tensor([[cos_a, sin_a], [-sin_a, cos_a]])
    # offset = torch.tensor([[-1, -1, 1, 1],  # width
    #                        [-1, 1, 1, -1]   # height
    #                        ])
    #
    # corners = center + (half_size * offset.T) @ k.T
    # return corners


class CornerLoss(nn.Module):
    """
    通过计算焦点之间的距离，来计算角点损失函数，越近损失值越小
    """
    def __init__(self):
        super(CornerLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 预测的边界框。可以取 shape=[B, N, 5]，或者 shape=[..., 5]
            target (torch.Tensor): 真实的边界框。可以取 shape=[B, N, 5]、shape=[B, 1, 5]，或者 shape=[..., 5]

        Returns:
            loss (torch.Tensor): 角点损失，shape=[B, N]
            distance (torch.Tensor): 角点距离，shape=[B, N, 4]
        """
        index = torch.tensor([[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]])
        pred_corners = obb_to_corners(pred)[..., index, :]          # [B, N, 5] -> [B, N, 4, 2] -> [B, N, 4, 4, 2]
        target_corners = obb_to_corners(target).unsqueeze(dim=-3)   # [B, N, 5] -> [B, N, 4, 2] -> [B, N, 1, 4, 2]

        # 计算角点之间的距离，也就是 sqrt(x^2 + y^2)
        # 其中 pred_corners - target_corners 得到的结果 shape=[B, N, 4, 4, 2]
        distance = torch.sqrt(((pred_corners - target_corners) ** 2).sum(dim=-1))   # [B, N, 4, 4]
        distance_sum = distance.sum(dim=-1)                                         # [B, N, 4]
        distance_sum_min, distance_sum_argmin = distance_sum.min(dim=-1)            # [B, N, 4], [B, N]

        index = (distance_sum_argmin                                # [B, N]
                 .unsqueeze(dim=-1)                                 # [B, N, 1]
                 .unsqueeze(dim=-1)                                 # [B, N, 1, 1]
                 .repeat_interleave(repeats=4, dim=-1))             # [B, N, 1, 4]
        # 得到 argmin 索引值对应的每个角点的 distance 值
        distance_min = torch.gather(distance, dim=-2, index=index)  # [B, N, 1, 4] -> [B, N, 4]

        # 距离总和
        loss, _ = distance_sum_min.min(dim=-1)                      # [B]
        # print(distance_sum_min.shape, loss.shape)

        return loss, distance_min                                   # [B], [B, N, 4]


if __name__ == '__main__':
    print(obb_to_corners(torch.tensor([0, 0, 2, 2, 0])))
    exit()
    a = torch.tensor([[0, 0, 2, 2, 0], [0, 0, 4, 4, 0]]).view(1, 1, -1, 5)
    b = torch.tensor([[1, 1, 2, 2, 0], [2, 2, 6, 6, 0]]).view(1, 1, -1, 5)
    c = CornerLoss()(a, b)
    print(c)

