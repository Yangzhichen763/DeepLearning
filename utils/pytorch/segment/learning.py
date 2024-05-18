import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.pytorch.segment import save_label_as_png
from utils.pytorch.segment.datasets import VOCSegmentationDataset


def train_model_in_single_epoch(model, training_loader, optimizer, criterion, device, writer=None, epoch=None):
    """
    训练模型一个 epoch.
    Args:
        model (torch.nn.Module): 用于训练的 PyTorch 模型.
        training_loader (DataLoader): 训练集的 DataLoader.
        optimizer (torch.optim.Optimizer): 训练优化器.
        criterion (torch.nn.Module): 损失函数.
        device (str | torch.device | int): 训练设备, 比如：'cpu' 或 'cuda'.
        writer (SummaryWriter): TensorBoard 日志记录器.
        epoch (int): 当前 epoch 数.
    """
    min_loss = float('inf')
    max_loss = float('-inf')
    total_loss = 0

    dataset_size = len(training_loader.dataset)
    dataset_batches = len(training_loader)
    batch_size = math.ceil(dataset_size / dataset_batches)

    model.train()  # 设置模型为训练模式
    with tqdm(
            total=dataset_size,
            unit='image') as pbar:
        pbar.set_description(f"Epoch {epoch} training")
        for (i, (image, ground_true)) in enumerate(training_loader):
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
            pbar.update(batch_size)
            # len(training_loader) 即训练集的总样本数 / batch_size
            # len(training_loader.dataset) 即训练集的总样本数

            if writer is not None:
                writer.add_scalar('./logs/tensorboard/loss', loss.item(), i_current_batch)

        pbar.close()

    if epoch is not None:
        tqdm.write(
            f"Epoch {epoch} training finished. "
            f"Min loss: {min_loss:.6f}, "
            f"Max loss: {max_loss:.6f}, "
            f"Avg loss: {total_loss / dataset_batches:.6f}")
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
        test_loader (DataLoader): 测试集的 DataLoader.
        criterion (torch.nn.Module): 损失函数.
        device (str | torch.device | int): 测试设备, 比如：'cpu' 或 'cuda'.
        writer (SummaryWriter): TensorBoard 日志记录器.
        epoch (int): 当前 epoch 数.
        validation_config (str): 验证配置, 目前支持 validation_configurations 中的部分配置.
    Returns:
        tuple: 平均损失和正确率.
    """
    if validation_config is None:
        validation_config = 'iou_correct'
    if validation_config not in validation_configurations:
        raise ValueError(f"Unsupported validation config: {validation_config}. "
                         f"Supported configs: {validation_configurations}.")

    dataset_size = len(test_loader.dataset)
    dataset_batches = len(test_loader)
    batch_size = math.ceil(dataset_size / dataset_batches)

    total_loss = 0
    correct = 0.
    model.eval()
    with tqdm(
            total=dataset_size,
            unit='image') as pbar, torch.no_grad():
        for (i, (images, true_masks)) in enumerate(test_loader):
            images, true_masks = images.to(device), true_masks.to(device)
            output = model(images)                                                  # 前向传播，获得输出
            loss = criterion(output, true_masks)                                    # 计算损失
            if validation_config == 'confusing_matrix':
                iou_list = calculate_iou(
                    outputs=output,
                    labels=true_masks)
                iou = torch.mean(iou_list)                                        # 计算正确率
            else:                                                                 # 计算 IoU
                iou_list = calculate_iou(
                    outputs=output,
                    labels=true_masks)
                iou = torch.mean(iou_list)                                        # 计算正确率

            total_loss += loss.item()
            correct += iou.item()

            # 打印测试进度
            pbar.update(batch_size)

            if i <= 10:
                save_label_as_png(
                    true_masks,
                    device,
                    VOCSegmentationDataset.color_map,
                    dir_path='./logs/targets',
                    file_name=f"target_{i}")
                save_label_as_png(
                    output,
                    device,
                    VOCSegmentationDataset.color_map,
                    dir_path='./logs/outputs',
                    file_name=f"output_{i}")

        pbar.close()

    average_loss = total_loss / dataset_batches
    accuracy = 100. * correct / dataset_batches
    tqdm.write(
        f"\nTest set: Average loss: {average_loss:.4f}, Accuracy: {int(correct * batch_size)}/{dataset_size} "
        f"({accuracy:.0f}%)\n")
    # 在测试集上，如果平均损失越大，说明模型过拟合
    # 1. 寻找更多数据
    # 2. 正则化
    # 3. 寻找合适的模型结构或更小的网络模型结构

    if (writer is not None) & (epoch is not None):
        writer.add_scalar('./logs/tensorboard/loss', average_loss, epoch)
        writer.add_scalar('./logs/tensorboard/accuracy', accuracy, epoch)

    return average_loss, accuracy


def calculate_iou(outputs, labels):
    """
    计算 IoU.
    Args:
        outputs (torch.Tensor): 模型输出，比如 [8, 21, 256, 256]
        labels (torch.Tensor): 标签，比如 [8, 256, 256]

    Returns:

    """
    num_classes = outputs.shape[1]
    outputs = outputs.argmax(dim=1)     # 将模型输出转换为预测的类别
    iou_list = []
    for classes in range(num_classes):  # 对每个类别计算IoU
        intersection = ((labels == classes) & (outputs == classes)).sum()
        union = ((labels == classes) | (outputs == classes)).sum()
        if union == 0:
            iou = 1.0  # 如果没有并集，则IoU为1
        else:
            iou = intersection / union
        iou_list.append(iou)
    return torch.tensor(iou_list)


# ----------------------------- 计算混淆矩阵 ---------------------------------

# 计算混淆矩阵
def _fast_hist(prediction, ground_true, num_classes):
    hist = torch.bincount(
        num_classes * ground_true +
        prediction, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


# 根据混淆矩阵计算 Accuracy, Accuracy_cls 和 Mean_IoU
def label_accuracy_score(predictions, ground_trues, num_classes):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = torch.zeros((num_classes, num_classes))

    # 计算混淆矩阵和平均准确率
    for l_true, l_pred in zip(ground_trues, predictions):
        hist += _fast_hist(l_pred.flatten(), l_true.flatten(), num_classes)
    accuracy = torch.diag(hist).sum() / hist.sum()

    # 计算类别平均准确率
    with torch.errstate(divide='ignore', invalid='ignore'):
        accuracy_class = torch.diag(hist) / hist.sum(dim=1)
    accuracy_class = torch.nanmean(accuracy_class)

    # 计算类别平均 IoU
    with torch.errstate(divide='ignore', invalid='ignore'):
        iou = torch.diag(hist) / (
                hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)
        )
    mean_iou = torch.nanmean(iou)

    # 计算全局准确率
    freq = hist.sum(dim=1) / hist.sum()
    return accuracy, accuracy_class, mean_iou, freq


if __name__ == '__main__':
    # 测试 save_label_as_image 函数
    _mask = torch.randint(0, 21, (21, 256, 256))
    for _i in range(21):
        _mask[_i, 100:150, 100:150] = _i
    save_label_as_png(_mask, device='cpu', dir_path='./logs/test', file_name='test')

