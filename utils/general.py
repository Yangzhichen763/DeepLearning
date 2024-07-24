import math
import os

from torch.cuda.amp import GradScaler
from tqdm import tqdm
import time as t
from time import time, sleep
import torch
import logging

from utils.torch import save, load
from utils.tensorboard import get_writer_by_name
from utils.log.info import print_

# 参数中 level 代表 INFO 即以上级别的日志信息才能被输出
logging.basicConfig(format="\n%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

checkpoint_relative_path = "./checkpoint"
current_time = t.strftime("%Y-%m-%d-%H-%M-%S")


__all__ = ["Trainer", "Validator", "Manager"]


def get_checkpoint_path(file_name: str):
    """
    通过文件名获取 checkpoint 完整路径
    """
    return os.path.join(checkpoint_relative_path, f"{file_name}.pt")


def save_states(model, file_name, path_unique=False, **kwargs):
    """
    Args:
        model (nn.Modules):
        file_name: 保存的文件名
        path_unique: 如果为 True，则路径如果重复，会使用序号改变文件名，使得文件名唯一
    """
    states = {
        'model': model.state_dict(),
        **kwargs
    }
    if kwargs.__contains__('optimizer'):
        states['optimizer'] = kwargs['optimizer'].state_dict()
    save_path = get_checkpoint_path(file_name)

    save.check_path(save_path, path_unique=path_unique)
    torch.save(states,  save_path)


def load_states(model, device, file_name, return_except=False, **kwargs):
    """
    加载模型、优化器等参数
    Args:
        model (nn.Module): 模型
        device: 设备
        file_name: 保存的文件名
        return_except: 如果为 True，则返回包括模型、优化器在内的所有参数，否则返回除了模型、优化器以外的其他参数
    Returns:
        加载的所有参数，通过形如 states['epoch'] 的方式访问
    """
    load_path = get_checkpoint_path(file_name)

    print_(f"Loading model from {load_path}...", end="")
    # 加载模型、优化器等参数
    states: dict = torch.load(load_path, map_location=device)
    model.load_state_dict(states['model'])
    if kwargs.__contains__('optimizer'):
        kwargs['optimizer'].load_state_dict(states['optimizer'])
    print_(f"\rModel {load_path} loaded.")

    if return_except:
        states.pop('model')
        for key in kwargs.keys():
            states.pop(key)
    return states


def buffer_dataloader(enumerate_obj):
    print_("Loading Data...", end="")
    start_time = time()
    datas = enumerate(enumerate_obj) if enumerate_obj is not None else None
    end_time = time()
    if end_time - start_time <= 0.01:
        sleep(0.5)
        print_("\r", end="")
    else:
        print_(f"\rTime to load data: {end_time - start_time:.2f}s")

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
        return value


class Trainer:
    # noinspection PyUnresolvedReferences
    """
        Example:
            从原本的：
            >>> dataset_size = len(train_loader.dataset)
            >>> dataset_batches = len(train_loader)
            >>> batch_size = math.ceil(dataset_size / dataset_batches)
            >>> for epoch in range(1, num_epochs + 1):
            >>>     total_loss = 0.
            >>>     num_items = 0
            >>>     for (i, (x, y)) in enumerate(data_loader):
            >>>         x = x.to(device)
            >>>
            >>>         loss = criterion(x, y)
            >>>         optimizer.zero_grad()
            >>>         loss.backward()
            >>>
            >>>         optimizer.step()
            >>>         total_loss += loss.item()
            >>>         num_items += batch_size
            >>>
            >>>     print(f"Average Loss: {total_loss / dataset_batches:5f}")
            >>>     torch.save(score_model.state_dict(), "ckpt.pth")

            改为：
            >>> trainer = Trainer(data_loader, optimizer)
            >>> manager = Manager(score_model, device)
            >>> for epoch in range(1, num_epochs + 1):
            >>>     datas = trainer.start(epoch)
            >>>     sde_model.train()
            >>>     for (i, (x, y)) in datas:
            >>>         x = x.to(device)
            >>>
            >>>         predict, loss = trainer.predict(sde_model, x, y, sde_model.loss_func)  # 预测和计算损失
            >>>         trainer.backward(loss)  # 反向传播
            >>>         trainer.step(i, loss)   # 更新参数
            >>>
            >>>     trainer.end()
        """
    def __init__(self, train_loader, optimizer, **kwargs):
        """
        Args:
            train_loader: 训练数据集
            optimizer: 优化器
            **kwargs: 包括 writer_enabled, writer, scaler，不设置 writer 时，默认使用当前时间作为 writer 名称
        """
        self.min_loss = None
        self.max_loss = None
        self.train_loader = train_loader
        self.optimizer = optimizer

        self.writer_enabled = kwargs.get('writer_enabled', True)
        if self.writer_enabled:
            self.writer = get_writer_by_name(current_time, "train") if kwargs.get('writer', None) is None else kwargs['writer']
        self.scaler = GradScaler() if kwargs.get('scaler', None) is None else kwargs['scaler']

    def __call__(self, epoch, **kwargs):
        """
        初始化训练过程，包括记录训练信息，初始化进度条等
        Args:
            epoch: 训练轮数
            **kwargs: 包括 scheduler
        """
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.total_loss = 0.
        self.dataset_size = len(self.train_loader.dataset)
        self.dataset_batches = len(self.train_loader)
        self.batch_size = math.ceil(self.dataset_size / self.dataset_batches)

        self.epoch = epoch

        # tqdm
        print_(f"\nEpoch {epoch}: ")
        if kwargs.get('scheduler', None) is not None:
            print_(f" - learning rate: {self.optimizer.param_groups[0]['lr']}")
        datas = buffer_dataloader(self.train_loader)
        self.process_bar = tqdm(   # 将 tqdm 放在加载 dataloader 之后，是因为防止进度条显示不正确
            total=self.dataset_size,
            desc=f"Training Epoch {self.epoch}",
            unit='image')

        return datas

    def start(self, epoch, **kwargs):
        """
        Args:
            epoch: 训练轮数
            **kwargs: 包括 scheduler
        """
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
            **kwargs: 包括 scheduler，输入其他的 kwarg 可以按照 key, value 的形式记录到 TensorBoard 中，并打印出来
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
            **kwargs: 包括 scheduler，输入其他的 kwarg 可以按照 key, value 的形式记录到 TensorBoard 中，并打印出来
        """
        self.process_bar.set_postfix()                                          # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%
        self.process_bar.close()

        # 打印损失信息
        print_(
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
        """
        Args:
            test_loader: 测试数据集
            **kwargs: 包括 writer_enabled, writer，不设置 writer 时，默认使用当前时间作为 writer 名称
        """
        self.test_loader = test_loader

        self.writer_enabled = kwargs.get('writer_enabled', True)
        if self.writer_enabled:
            self.writer = get_writer_by_name(current_time, "test") if kwargs.get('writer', None) is None else kwargs['writer']

    def __call__(self, epoch, optimizer, **kwargs):
        self.total_loss = 0.
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
            **kwargs: 输入其他的 kwarg 可以按照 key, value 的形式记录到 TensorBoard 中，并打印出来
        """
        loss = get_value(loss)
        self.total_loss += loss
        self.correct += get_value(correct)

        # 打印训练进度
        self.process_bar.set_postfix(loss=loss)
        self.process_bar.update(self.batch_size)

    def end(self, **kwargs):
        """
        Args:
            **kwargs: 输入其他的 kwarg 可以按照 key, value 的形式记录到 TensorBoard 中，并打印出来
        """
        self.process_bar.set_postfix()                                          # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%
        self.process_bar.close()

        # 计算平均损失
        average_loss = self.total_loss / self.dataset_batches
        print_(f"Average loss: {average_loss:.4f}")
        # 计算准确率
        correct = self.correct if kwargs.get('correct', None) is None else kwargs['correct']
        total = self.dataset_size if kwargs.get('total', None) is None else kwargs['total']
        accuracy = 100. * correct / total
        print_(f"Accuracy: {correct:.0f}/{total:.0f} ({accuracy:.2f}%)")
        # 打印其他信息
        output_list = []
        for key, value in kwargs.items():
            if is_value(value) and key not in ['correct', 'total']:
                print_(f"{key}: {get_value(value):.4f}")
                output_list.append(get_value(value))

        # 记录到 TensorBoard
        if self.writer_enabled:
            self.writer.add_scalar("test_loss", average_loss, self.epoch)
            self.writer.add_scalar("test_accuracy", accuracy, self.epoch)
            for key, value in kwargs.items():
                if is_value(value):
                    self.writer.add_scalar(f"test_{key}", get_value(value), self.epoch)

        return average_loss, accuracy, *output_list


class Manager:
    def __init__(self, model, device, **kwargs):
        """
        Args:
            model: 模型
            device: 设备
            **kwargs: 包括 optimizer, writer，不设置 writer 时，默认使用当前时间作为 writer 名称
        """
        self.best_accuracy = 0.
        self.last_accuracy = 0.
        self.model = model
        if kwargs.__contains__('optimizer'):
            self.optimizer = kwargs['optimizer']
        self.device = device

        self.writer = kwargs.get('writer') if kwargs.get("writer", None) is not None else None
        self.url = os.path.basename(self.writer.log_dir) \
            if self.writer is not None \
            else current_time

    def resume_checkpoint(self, file_name=None, **kwargs):
        """
        加载上一次的模型训练数据
        Args:
            file_name: 要加载的 checkpoint 文件名，如果为 None，则加载最新的 checkpoint，文件名为 last
            **kwargs: 更新模型的参数数据，可以包括 optimizer 等
        Returns:
            加载的数据，通过形如 states['epoch'] 的方式访问
        """
        if file_name is None:
            file_name = "last"

        states = load_states(self.model, self.device, file_name, **kwargs)
        return states

    def update_checkpoint(self, accuracy, **kwargs):
        """
        更新准确率以及保存模型
        Args:
            accuracy: 用于判断是否更新最佳模型，以及保存模型的准确率
            **kwargs: 更新模型的参数数据，可以包括 optimizer 等
        """
        # 更新准确率以及保存模型
        self.last_accuracy = get_value(accuracy)
        if self.last_accuracy > self.best_accuracy:
            self.best_accuracy = self.last_accuracy
            save_states(self.model, "best", **kwargs)
        save_states(self.model, f"ckpt_{self.url}", path_unique=True, **kwargs)
        save_states(self.model, "last", **kwargs)

    def summary(self, **kwargs):
        """
        Args:
            **kwargs: 输入其他的 kwarg 可以按照 key, value 的形式记录到 TensorBoard 中，并打印出来
        """
        # 打印总结信息
        print_(f"\nSummary: ")
        print_(f" - [Best Accuracy]: {self.best_accuracy:.2f}%  [Last Accuracy]: {self.last_accuracy:.2f}%")
        for key, value in kwargs.items():
            print_(f" - [{key}]: {get_value(value)}")

    def save_checkpoint(self, **kwargs):
        """
        保存最新和最好的模型
        Args:
            **kwargs: 更新模型的参数数据，可以包括 optimizer 等
        """
        save_states(self.model, f"last_{self.url}", **kwargs)
        load_states(self.model, self.device, "best", **kwargs)
        save_states(self.model, f"best_{self.url}", **kwargs)

