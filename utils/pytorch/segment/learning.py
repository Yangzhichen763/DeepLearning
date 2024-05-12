import time
from math import floor

import torch
import numpy as np


def train_model_in_single_epoch(model, training_loader, optimizer, criterion, device, writer=None, epoch=None):
    """
    训练模型一个 epoch.
    Args:
        model (torch.nn.Module): 用于训练的 PyTorch 模型.
        training_loader (torch.classify_utils.data.DataLoader): 训练集的 DataLoader.
        optimizer (torch.optim.Optimizer): 训练优化器.
        criterion (torch.nn.Module): 损失函数.
        device (str | torch.device | int): 训练设备, 比如：'cpu' 或 'cuda'.
        writer (SummaryWriter): TensorBoard 日志记录器.
        epoch (int): 当前 epoch 数.
    """
    print('_' * 30)
    print(f"Epoch {epoch} training started.")

    start_time = time.time()

    min_loss = float('inf')
    max_loss = float('-inf')
    total_loss = 0
    model.train()  # 设置模型为训练模式
    for i, (image, ground_true) in enumerate(training_loader):
        image, ground_true = image.to(device), ground_true.to(device)
        optimizer.zero_grad()                       # 清空梯度
        output = model(image)                       # 前向传播，获得输出
        loss = criterion(output, ground_true)       # 计算损失
        loss.backward()                             # 反向传播
        optimizer.step()                            # 更新参数

        # 记录损失最小值和最大值以及总损失
        min_loss = min(min_loss, loss.item())
        max_loss = max(max_loss, loss.item())
        total_loss += loss.item()

        # 打印训练进度
        i_current_batch = i + 1
        if floor(100. * i_current_batch / len(training_loader)) > floor(100. * i / len(training_loader)):
            print(
                f"[{i_current_batch * len(image)}/{len(training_loader.dataset)} "
                f"({floor(100. * i_current_batch / len(training_loader)):.0f}%)]\tLoss: {loss.item():.6f}")
        # len(training_loader) 即训练集的总样本数 / batch_size
        # len(training_loader.dataset) 即训练集的总样本数

        if writer is not None:
            writer.add_scalar('./logs_tensorboard/loss', loss.item(), i_current_batch)

    if epoch is not None:
        print(f"Epoch {epoch} training finished. "
              f"Min loss: {min_loss:.6f}, "
              f"Max loss: {max_loss:.6f}, "
              f"Avg loss: {total_loss / len(training_loader):.6f}")

    end_time = time.time()
    print(f"Epoch {epoch} training time: {end_time - start_time:.2f}s")
    # 在训练集上，如果最大最小损失相差太大，说明学习率过大
    # 在训练集上，如果平均损失越大，说明模型欠拟合
    # 1. 扩大训练集规模
    # 2. 寻找更好的模型结构或更大的网络模型结构
    # 3. 花费更多时间训练


validation_configurations = {
    'iou_correct',          # 使用 IoU 作为正确率的评估指标
    'confusing_matrix',     # 使用混淆矩阵作为正确率的评估指标
}


def validate_model(
        model,
        test_loader,
        criterion,
        device,
        writer=None,
        epoch=None,
        validation_config=None):
    """
    测试模型.
    Args:
        model (torch.nn.Module): 用于测试的 PyTorch 模型.
        test_loader (torch.classify_utils.data.DataLoader): 测试集的 DataLoader.
        criterion (torch.nn.Module): 损失函数.
        device (str | torch.device | int): 测试设备, 比如：'cpu' 或 'cuda'.
        writer (SummaryWriter): TensorBoard 日志记录器.
        epoch (int): 当前 epoch 数.
        validation_config (str): 验证配置, 目前支持 validation_configurations 中的部分配置.
    Returns:
        tuple: 总损失函数值和正确率.
    """
    total_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for (images, true_masks) in test_loader:
            images, true_masks = images.to(device), true_masks.to(device)
            output = model(images)                                                  # 前向传播，获得输出
            total_loss += criterion(output, true_masks).item()                      # 计算损失
            if validation_config == 'iou_correct':
                iou_list = calculate_iou(output, true_masks, num_classes=2)             # 计算 IoU
                correct += np.mean(iou_list)                                            # 计算正确率
            else:
                prediction = output.argmax(dim=1, keepdim=True)                         # 预测类别
                correct += prediction.eq(true_masks.view_as(prediction)).sum().item()   # 计算正确率
    total_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:.0f}%)\n")
    # 在测试集上，如果平均损失越大，说明模型过拟合
    # 1. 寻找更多数据
    # 2. 正则化
    # 3. 寻找合适的模型结构或更小的网络模型结构

    if (writer is not None) & (epoch is not None):
        writer.add_scalar('./logs_tensorboard/loss', total_loss, epoch)
        writer.add_scalar('./logs_tensorboard/accuracy', accuracy, epoch)

    return total_loss, accuracy


def calculate_iou(outputs, labels, num_classes):
    outputs = np.argmax(outputs, axis=1)  # 将模型输出转换为预测的类别
    iou_list = []
    for cls in range(num_classes):  # 对每个类别计算IoU
        intersection = np.logical_and(labels == cls, outputs == cls).sum()
        union = np.logical_or(labels == cls, outputs == cls).sum()
        if union == 0:
            iou = 1.0  # 如果没有交集，则IoU为1
        else:
            iou = intersection / union
        iou_list.append(iou)
    return iou_list


# ----------------------------- 计算混淆矩阵 ---------------------------------

# 计算混淆矩阵
def _fast_hist(ground_true, prediction, num_classes):
    mask = (ground_true >= 0) & (ground_true < num_classes)
    hist = np.bincount(
        num_classes * ground_true[mask].astype(int) +
        prediction[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


# 根据混淆矩阵计算 Accuracy, Accuracy_cls 和 Mean_IoU
def label_accuracy_score(ground_trues, predictions, num_classes):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((num_classes, num_classes))

    # 计算混淆矩阵和平均准确率
    for l_true, l_pred in zip(ground_trues, predictions):
        hist += _fast_hist(l_true.flatten(), l_pred.flatten(), num_classes)
    accuracy = np.diag(hist).sum() / hist.sum()

    # 计算类别平均准确率
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy_class = np.diag(hist) / hist.sum(axis=1)
    accuracy_class = np.nanmean(accuracy_class)

    # 计算类别平均 IoU
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iou = np.nanmean(iou)

    # 计算全局准确率
    freq = hist.sum(axis=1) / hist.sum()
    return accuracy, accuracy_class, mean_iou, freq
