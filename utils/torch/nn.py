import torch

from utils.log.info import print_


def flatten_(self, start_dim=0, end_dim=-1):
    """
    相当于 view，但是更加灵活
    \nstart_dim=0, end_dim=-1 表示将所有维度压缩，[A, B, C, D] -> [A*B*C*D]
    \nstart_dim=1, end_dim=-2 表示将第1维到倒数第2维压缩，[A, B, C, D] -> [A, B*C, D]
    Args:
        start_dim: 起始维度
        end_dim: 结束维度，包括在内
    """
    if end_dim == -1:
        return self.view(self.shape[:start_dim] + (-1,))
    else:
        return self.view(self.shape[:start_dim] + (-1,) + self.shape[end_dim+1:])


torch.Tensor.flatten_ = flatten_


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 5)
    print_(x.flatten_(start_dim=-4, end_dim=-2).shape)
