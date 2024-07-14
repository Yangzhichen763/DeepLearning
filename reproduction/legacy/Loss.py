import numpy as np


def loss_hinge_iterated(x, y, weight):
    """
    非向量化的 SVM 损失函数，用于计算单个样例 (x, y) 的多类 SVM 损失
    :param x: 表示图像的列向量（例：图像大小 32x32x3，则 x 的形状为 3072x1）
    :param y: 正确的分类标签（例：分类总数为 10，则 y 只能在 0 ~ 9 内取值）
    :param weight: 权重矩阵（例：x 形状为 3072x1，分类总数为 10，则 weight 形状为 10x3072）
    :return: 损失值
    """
    delta = 1.0                         # 损失容错值（例：当前类别的 score - 目标类别的 score > 1，则需要对当前分类进行损失计算）
    scores = weight.dot(x)              # 得分数组（例：形状为 10x1）
    correct_class_score = scores[y]     # 目标类别得到的分数
    n_class = weight.shape[0]           # 分类数量（例：值为 10）
    loss_y = 0.0
    for i in range(n_class):
        # 如果当前类别 i 和目标类别 y 一致，则跳过计算当前损失
        if i == y:
            continue
        # 累计每个类别 i 的损失
        loss_y += max(0, scores[i] - correct_class_score + delta)
    return loss_y


def loss_hinge_vectorized(x, y, weight):
    """
    半向量化的 SVM 损失函数，用于计算单个样例 (x, y) 的多类 SVM 损失，以向量的方式加速计算过程
    :param x: 表示图像的列向量（例：图像大小 32x32x3，则 x 的形状为 3072x1）
    :param y: 正确的分类标签（例：分类总数为 10，则 y 只能在 0 ~ 9 内取值）
    :param weight: 权重矩阵（例：x 形状为 3072x1，分类总数为 10，则 weight 形状为 10x3072）
    :return: 损失值
    """
    delta = 1.0                                             # 损失容错值（例：当前类别的 score - 目标类别的 score > 1，则需要对当前分类进行损失计算）
    scores = weight.dot(x)                                  # 得分数组（例：形状为 10x1）
    margins = np.maximum(0, scores - scores[y] + delta)     # 每个类别的损失值组成的行向量
    margins[y] = 0                                          # 将目标类别 y 的损失值设置为 0，即不计入类别 y 的损失值
    loss_y = np.sum(margins)                                  # 计算目标类别 y 的损失值
    return loss_y


def loss_softmax_vectorized(x, y, weight):
    """
    归一化损失函数，计算所有分类的损失
    :param x: 表示图像的列向量（例：图像大小 32x32x3，则 x 的形状为 3072x1）
    :param y: 正确的分类标签（例：分类总数为 10，则 y 只能在 0 ~ 9 内取值）
    :param weight: 权重矩阵（例：x 形状为 3072x1，分类总数为 10，则 weight 形状为 10x3072）
    :return: 损失值
    """
    scores = weight.dot(x)                                      # 得分数组（例：形状为 10x1）
    scores -= np.max(scores)                                    # 避免指数爆炸，超出上界（超出下界，归 0）
    probability = np.exp(scores[y]) / np.sum(np.exp(scores))     # 计算每个分类的归一化概率
    loss_y = -np.log(probability)
    return loss_y


def evaluate_numerical_gradient(func_loss, weight):
    """
    数值梯度法求 loss 关于 weight 的梯度，使用中心差值公式计算
    当自己写梯度计算时，可以用来检验
    :param func_loss: 损失值矩阵计算函数
    :param weight: 当前的权重矩阵（例：x 形状为 3072x1，分类总数为 10，则 weight 形状为 10x3072）
    :return: 梯度值
    """
    x = weight
    dx = 0.0001                                         # 变化量，用于近似 dx 计算梯度
    f_x_plus_dx = func_loss(x + dx)                     # 在 x + dx 处的损失值
    f_x_minus_dx = func_loss(x - dx)                    # 在 x - dx 处的损失值
    gradiant = (f_x_plus_dx - f_x_minus_dx) / (2 * dx)  # 梯度矩阵
    return gradiant


def evaluate_analytical_gradient(func_loss, weight):
    """
    分析梯度法求 loss 关于 weight 的梯度，即直接使用导数公式计算
    :param func_loss: 损失值矩阵计算函数
    :param weight: 当前的权重矩阵（例：x 形状为 3072x1，分类总数为 10，则 weight 形状为 10x3072）
    :return: 梯度值
    """
    pass





