from torch.cuda.amp import GradScaler
from tqdm import tqdm
import torch

from utils.pytorch.dataset import buffer_dataloader
from utils.tensorboard import get_writer


class Trainer:
    def __init__(self, train_loader, **kwargs):
        self.train_loader = train_loader

        self.writer = get_writer() if kwargs.get('writer', None) is None else kwargs['writer']
        self.scaler = GradScaler() if kwargs.get('scaler', None) is None else kwargs['scaler']

    def __call__(self, epoch, optimizer, **kwargs):
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.total_loss = 0
        self.dataset_size = len(self.train_loader.dataset)
        self.dataset_batches = len(self.train_loader)

        self.epoch = epoch
        self.optimizer = optimizer

        tqdm.write(f"\nEpoch {epoch}: ")
        if kwargs.get('scheduler', None) is not None:
            tqdm.write(f" - learning rate: {self.optimizer.param_groups[0]['lr']}")
        datas = buffer_dataloader(self.train_loader)

        self.process_bar = tqdm(   # 将 tqdm 放在加载 dataloader 之后，是因为防止进度条显示不正确
            total=self.dataset_size,
            desc=f"Training Epoch {self.epoch}",
            unit='image')

        return datas

    def backward(self, loss, **kwargs):
        self.optimizer.zero_grad()                      # 清空梯度
        self.scaler.scale(loss).backward()              # 反向传播
        self.scaler.step(self.optimizer)                # 更新参数
        self.scaler.update()                            # 更新 GradScaler
        if kwargs.get('scheduler', None) is not None:
            scheduler: torch.optim.lr_scheduler.LRScheduler = kwargs['scheduler']
            scheduler.step()                  # 更新学习率

    def step(self, i, loss, **kwargs):
        # 记录损失最小值和最大值以及总损失
        self.min_loss = min(self.min_loss, loss.item())
        self.max_loss = max(self.max_loss, loss.item())
        self.total_loss += loss.item()

        # 打印训练进度
        i_current = self.epoch * self.dataset_batches + i
        self.process_bar.set_postfix()                                          # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%

        if self.writer is not None:
            self.writer.add_scalar(
                f"train_loss",
                loss.item(),
                i_current)
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and (value.dim() == 0 or value.dim() == 1):
                    self.writer.add_scalar(
                        f"train_{key}",
                        value.item(),
                        i_current)
            if kwargs.get('scheduler_batch', None) is not None:
                self.writer.add_scalar(
                    "train_lr",
                    self.optimizer.param_groups[0]['lr'],
                    i_current)

    def end(self, **kwargs):
        self.process_bar.set_postfix()                                          # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%

        tqdm.write(
            f"Min-Avg-Max loss: [{self.min_loss:.4f}, {self.total_loss / self.dataset_batches:.4f}, {self.max_loss:.4f}]")

        if kwargs.get('scheduler', None) is not None:
            scheduler: torch.optim.lr_scheduler.LRScheduler = kwargs['scheduler']
            scheduler.step()  # 更新学习率
            self.writer.add_scalar(
                "train_lr",
                self.optimizer.param_groups[0]['lr'],
                self.epoch)


class Validator:
    def __init__(self, test_loader, **kwargs):
        self.test_loader = test_loader

        self.writer = get_writer() if kwargs.get('writer', None) is None else kwargs['writer']

    def __call__(self, epoch, optimizer, **kwargs):
        self.total_loss = 0
        self.correct = 0.
        self.dataset_size = len(self.test_loader.dataset)
        self.dataset_batches = len(self.test_loader)

        self.epoch = epoch
        self.optimizer = optimizer

        datas = buffer_dataloader(self.test_loader)
        self.process_bar = tqdm(   # 将 tqdm 放在加载 dataloader 之后，是因为防止进度条显示不正确
            total=self.dataset_size,
            desc=f"Testing Epoch {self.epoch}",
            unit='image')

        return datas

    def step(self, i, loss, correct, **kwargs):
        # 打印训练进度
        self.process_bar.set_postfix()                                          # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%

        self.total_loss += loss.item()
        self.correct += correct.item()

    def end(self, **kwargs):
        self.process_bar.set_postfix()                  # 清空进度条备注
        self.process_bar.update(self.process_bar.total - self.process_bar.n)    # 防止进度条超过 100%

        average_loss = self.total_loss / self.dataset_batches
        accuracy = 100. * self.correct / self.dataset_size
        tqdm.write(
            f"Average loss: {average_loss:.4f}, "
            f"Accuracy: {self.correct:.0f}/{self.dataset_size} ({accuracy:.2f}%)")

        self.writer.add_scalar("test_loss", average_loss, self.epoch)
        self.writer.add_scalar("accuracy", accuracy, self.epoch)