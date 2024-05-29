import torch
import torch.nn as nn
import torch.nn.functional as F


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


def rotated_rect_to_corners(rotated_rect):
    """
    包围盒转化为角点，即 xywhr -> xy sigma
    Args:
        rotated_rect (torch.Tensor | list(torch.Tensor)): torch.tensor([x, y, w, h, angle])，其中 angle 为弧度制

    Returns:
        (torch.tensor): torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        顺时针方向返回角点位置
        [x, y, w, h, angle] -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
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
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)           # [num_rect, 1]
    k = (torch
         .stack(tensors=[cos_a, -sin_a, sin_a, cos_a], dim=-1)  # [num_rect, 1, 4]
         .reshape(-1, 2, 2))                                    # [num_rect, 2, 2]
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
    # cos_a, sin_a = math.cos(angle), math.sin(angle)
    # k = torch.tensor([[cos_a, sin_a], [-sin_a, cos_a]])
    # offset = torch.tensor([[-1, -1, 1, 1],  # width
    #                        [-1, 1, 1, -1]   # height
    #                        ])
    #
    # corners = center + (half_size * offset.T) @ k.T
    # return corners


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
