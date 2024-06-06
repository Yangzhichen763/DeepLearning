import math

from torch.cuda.amp import GradScaler
from tqdm import tqdm
from time import time, sleep
import torch
import logging

from utils.tensorboard import get_writer


# 参数中 level 代表 INFO 即以上级别的日志信息才能被输出
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")


def buffer_dataloader(enumerate_obj):
    tqdm.write("Loading Data...", end="")
    start_time = time()
    datas = enumerate(enumerate_obj) if enumerate_obj is not None else None
    end_time = time()
    if end_time - start_time <= 0.01:
        sleep(0.5)
        tqdm.write("\r", end="")
    else:
        tqdm.write(f"\rTime to load data: {end_time - start_time:.2f}s")

    return datas


def is_value(value):
    """
    判断 value 是否为 torch.Tensor 或者标量
    """
    return isinstance(value, (int, float, torch.Tensor))


def get_value(value):
    """
    获取 torch.Tensor 或者标量的值
    """
    if isinstance(value, torch.Tensor) and (value.dim() == 0 or value.dim() == 1):
        return value.item()
    elif isinstance(value, (int, float)):
        return value
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")


class Trainer:
    def __init__(self, train_loader, optimizer, **kwargs):
        self.min_loss = None
        self.max_loss = None
        self.train_loader = train_loader
        self.optimizer = optimizer

        self.writer_enabled = kwargs.get('writer_enabled', True)
        if self.writer_enabled:
            self.writer = get_writer() if kwargs.get('writer', None) is None else kwargs['writer']
        self.scaler = GradScaler() if kwargs.get('scaler', None) is None else kwargs['scaler']

    def __call__(self, epoch, **kwargs):
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.total_loss = 0
        self.dataset_size = len(self.train_loader.dataset)
        self.dataset_batches = len(self.train_loader)
        self.batch_size = math.ceil(self.dataset_size / self.dataset_batches)

        self.epoch = epoch

        logging.info(f"\nEpoch {epoch}: ")
        if kwargs.get('scheduler', None) is not None:
            tqdm.write(f" - learning rate: {self.optimizer.param_groups[0]['lr']}")
        datas = buffer_dataloader(self.train_loader)

        self.process_bar = tqdm(   # 将 tqdm 放在加载 dataloader 之后，是因为防止进度条显示不正确
            total=self.dataset_size,
            desc=f"Training Epoch {self.epoch}",
            unit='image')

        return datas

    def start(self, epoch, **kwargs):
        return self.__call__(epoch, **kwargs)

    def backward(self, loss, **kwargs):
        """
        反向传播封装，使用 GradScaler 进行自动混合精度
        Args:
            loss: 损失值
            **kwargs: 包括 scheduler

        Returns:

        """
        self.optimizer.zero_grad()                      # 清空梯度
        self.scaler.scale(loss).backward()              # 反向传播
        self.scaler.step(self.optimizer)                # 更新参数
        self.scaler.update()                            # 更新 GradScaler
        if kwargs.get('scheduler', None) is not None:
            scheduler: torch.optim.lr_scheduler.LRScheduler = kwargs['scheduler']
            scheduler.step()                  # 更新学习率

    def step(self, i, loss, **kwargs):
        """
        Args:
            i: 批次序号，也就是当前 batch 的索引
            loss: 损失值
            **kwargs: 包括 scheduler

        Returns:

        """
        # 记录损失最小值和最大值以及总损失
        loss = get_value(loss)
        self.min_loss = min(self.min_loss, loss)
        self.max_loss = max(self.max_loss, loss)
        self.total_loss += loss

        # 打印训练进度
        i_current = self.epoch * self.dataset_batches + i
        self.process_bar.set_postfix(loss=loss)
        self.process_bar.update(self.batch_size)

        if self.writer_enabled:
            self.writer.add_scalar(f"train_loss", loss, i_current)
            for key, value in kwargs.items():
                if is_value(value):
                    self.writer.add_scalar(f"train_{key}", get_value(value), self.epoch)
            if kwargs.get('scheduler', None) is not None:
                self.writer.add_scalar("train_lr", self.optimizer.param_groups[0]['lr'], i_current)

    def end(self, **kwargs):
        """
        训练收尾，打印训练信息，更新学习率
        Args:
            **kwargs: 包括 scheduler
        """
        self.process_bar.set_postfix()                                          # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%
        self.process_bar.close()

        # 打印损失信息
        tqdm.write(
            f"Min-Avg-Max loss: [{self.min_loss:.4f}, {self.average_loss:.4f}, {self.max_loss:.4f}]")

        # 更新学习率
        if kwargs.get('scheduler', None) is not None:
            scheduler: torch.optim.lr_scheduler.LRScheduler = kwargs['scheduler']
            scheduler.step()  # 更新学习率

            if self.writer_enabled:
                self.writer.add_scalar("train_lr", self.optimizer.param_groups[0]['lr'], self.epoch)

    @property
    def average_loss(self):
        return self.total_loss / self.dataset_batches

    @staticmethod
    def predict(model, images, targets, criterion):
        """
        简单的标签预测
        Args:
            model: 模型
            images: 输入图像
            targets: 标签
            criterion: 损失函数

        Returns: 模型预测结果，损失值
        """
        with torch.cuda.amp.autocast():
            predict = model(images)
            loss = criterion(predict, targets)
        return predict, loss


class Validator:
    def __init__(self, test_loader, **kwargs):
        self.test_loader = test_loader

        self.writer_enabled = kwargs.get('writer_enabled', True)
        if self.writer_enabled:
            self.writer = get_writer() if kwargs.get('writer', None) is None else kwargs['writer']

    def __call__(self, epoch, optimizer, **kwargs):
        self.total_loss = 0
        self.correct = 0.
        self.dataset_size = len(self.test_loader.dataset)
        self.dataset_batches = len(self.test_loader)
        self.batch_size = math.ceil(self.dataset_size / self.dataset_batches)

        self.epoch = epoch
        self.optimizer = optimizer

        datas = buffer_dataloader(self.test_loader)
        self.process_bar = tqdm(   # 将 tqdm 放在加载 dataloader 之后，是因为防止进度条显示不正确
            total=self.dataset_size,
            desc=f"Testing Epoch {self.epoch}",
            unit='image')

        return datas

    def start(self, epoch, optimizer, **kwargs):
        """
        需要以 (i, (images, labels)) 的方式遍历
        Args:
            epoch:
            optimizer:
            **kwargs:

        Returns:

        """
        return self.__call__(epoch, optimizer, **kwargs)

    def step(self, i, loss, correct, **kwargs):
        """
        打印测试进度，累加损失和正确率
        Args:
            i: 批次序号，也就是当前 batch 的索引
            loss: 损失值
            correct: 准确率
            **kwargs:
        """
        loss = get_value(loss)
        self.total_loss += loss
        self.correct += get_value(correct)

        # 打印训练进度
        self.process_bar.set_postfix(loss=loss)
        self.process_bar.update(self.batch_size)

    def end(self, **kwargs):
        self.process_bar.set_postfix()                                          # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%
        self.process_bar.close()

        # 计算平均损失
        average_loss = self.total_loss / self.dataset_batches
        tqdm.write(f"Average loss: {average_loss:.4f}")
        # 计算准确率
        correct = self.correct if kwargs.get('correct', None) is None else kwargs['correct']
        total = self.dataset_size if kwargs.get('total', None) is None else kwargs['total']
        accuracy = 100. * correct / total
        tqdm.write(f"Accuracy: {correct:.0f}/{total:.0f} ({accuracy:.2f}%)")
        # 打印其他信息
        output_list = []
        for key, value in kwargs.items():
            if is_value(value) and key not in ['correct', 'total']:
                tqdm.write(f"{key}: {get_value(value):.4f}")
                output_list.append(get_value(value))

        # 记录到 TensorBoard
        if self.writer_enabled:
            self.writer.add_scalar("test_loss", average_loss, self.epoch)
            self.writer.add_scalar("test_accuracy", accuracy, self.epoch)
            for key, value in kwargs.items():
                if is_value(value):
                    self.writer.add_scalar(f"test_{key}", get_value(value), self.epoch)

        return average_loss, accuracy, *output_list
