import time
from math import floor

import torch


def train_model_in_single_epoch(model, train_loader, optimizer, criterion, device, writer=None, epoch=None):
    """
    训练模型一个 epoch.
    Args:
        model (torch.nn.Module): 用于训练的 PyTorch 模型.
        train_loader (torch.classify_utils.data.DataLoader): 训练集的 DataLoader.
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
    for i_batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # 清空梯度
        output = model(data)                # 前向传播，获得输出
        loss = criterion(output, target)    # 计算损失
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新参数

        # 记录损失最小值和最大值以及总损失
        min_loss = min(min_loss, loss.item())
        max_loss = max(max_loss, loss.item())
        total_loss += loss.item()

        # 打印训练进度
        i_current_batch = i_batch + 1
        if floor(100. * i_current_batch / len(train_loader)) > floor(100. * i_batch / len(train_loader)):
            print(
                f"[{i_current_batch * len(data)}/{len(train_loader.dataset)} "
                f"({floor(100. * i_current_batch / len(train_loader)):.0f}%)]\tLoss: {loss.item():.6f}")

        if writer is not None:
            writer.add_scalar('./logs_tensorboard/loss', loss.item(), i_current_batch)

    if epoch is not None:
        print(f"Epoch {epoch} training finished. "
              f"Min loss: {min_loss:.6f}, "
              f"Max loss: {max_loss:.6f}, "
              f"Avg loss: {total_loss / len(train_loader):.6f}")

    end_time = time.time()
    print(f"Epoch {epoch} training time: {end_time - start_time:.2f}s")
    # 在训练集上，如果最大最小损失相差太大，说明学习率过大
    # 在训练集上，如果平均损失越大，说明模型欠拟合
    # 1. 扩大训练集规模
    # 2. 寻找更好的模型结构或更大的网络模型结构
    # 3. 花费更多时间训练


def validate_model(model, test_loader, criterion, device, writer=None, epoch=None):
    """
    测试模型.
    Args:
        model (torch.nn.Module): 用于测试的 PyTorch 模型.
        test_loader (torch.classify_utils.data.DataLoader): 测试集的 DataLoader.
        criterion (torch.nn.Module): 损失函数.
        device (str | torch.device | int): 测试设备, 比如：'cpu' 或 'cuda'.
        writer (SummaryWriter): TensorBoard 日志记录器.
        epoch (int): 当前 epoch 数.
    Returns:
        tuple: 总损失函数值和正确率.
    """
    total_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)                                                # 前向传播，获得输出
            total_loss += criterion(output, target).item()                      # 计算损失
            prediction = output.argmax(dim=1, keepdim=True)                     # 预测类别
            correct += prediction.eq(target.view_as(prediction)).sum().item()   # 计算正确率
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
