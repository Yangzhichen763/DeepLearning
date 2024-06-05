import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.general import Trainer, Validator
from utils.tensorboard import *

from model import (UNet, UNetCustom)
from model import (UNet_custom_light)
from lately.segment_utils import get_transform, train_and_validate
from utils.pytorch import *
from utils.pytorch.dataset import buffer_dataloader
from utils.pytorch.segment.datasets import CarvanaDataset, VOCSegmentationDataset


def carvana(test_model=False):
    def train_epoch():
        datas = trainer.start(epoch, scheduler=scheduler)

        model.train()
        for (i, (images, labels)) in datas:
            images, labels = images.to(device), labels.unsqueeze(1).float().to(device)

            predict, loss = trainer.predict(model, images, labels, criterion)   # 预测和计算损失
            trainer.backward(loss)                                              # 反向传播
            trainer.step(i, loss)                                               # 更新参数

        trainer.end(scheduler=scheduler)
        return trainer.average_loss

    def validate_epoch():
        datas = validator.start(epoch, optimizer)

        num_correct = 0
        num_pixels = 0
        dice_score = 0.0

        model.eval()
        with torch.no_grad():
            for (i, (images, labels)) in datas:
                images, labels = images.to(device), labels.unsqueeze(1).float().to(device)

                predict = model(images)

                # 计算损失
                loss = criterion(predict, labels)

                # 预测和计算准确率和 Dice 系数
                predict = torch.sigmoid(predict)
                predict = (predict > 0.5).float()
                num_correct += (predict == labels).sum()
                num_pixels += torch.numel(predict)
                dice_score += ((2 * (predict * labels).sum())
                               / (2 * (predict * labels).sum()
                               + ((predict * labels) < 1).sum()))
                # if test_model:
                save.tensor_to_image(predict, file_name="pred", batch=i)

                validator.step(i, loss, num_correct / num_pixels)

        average_loss, accuracy, dice = validator.end(
                correct=num_correct, total=num_pixels,
                dice=dice_score / validator.dataset_batches)
        return average_loss, accuracy, dice

    # 超参数设置
    batch_size = 32
    num_workers = 2
    num_epochs = 50
    learning_rate = 3e-4
    weight_decay = 0.0001
    momentum = 0.9
    scheduler_step_size = 5
    scheduler_gamma = 0.71
    image_size = (160, 240)

    # 部署 GPU 设备
    device = assert_on_cuda()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_transform = A.Compose([
        A.Resize(*image_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ],)
    val_transform = A.Compose([
        A.Resize(*image_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ],)

    # 加载数据集
    if not test_model:
        train_dataset = CarvanaDataset(transform=train_transform)
        train_dataset = Subset(train_dataset, range(0, 800))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, persistent_workers=True, pin_memory=True)
    val_dataset = CarvanaDataset(transform=val_transform)
    val_dataset = Subset(val_dataset, range(0, 200))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True, pin_memory=True)

    # 定义模型
    model = UNet_custom_light(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    if not test_model:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # 训练模型
        trainer = Trainer(train_loader, optimizer, writer_enabled=False)
        validator = Validator(val_loader)
        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch()
            val_loss, val_accuracy, val_dice = validate_epoch()
            scheduler.step()
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f"models/{model.__class__.__name__}.pt")
    else:
        # 加载模型
        model.load_state_dict(torch.load(f"models/{model.__class__.__name__}.pt"))

        # 测试模型
        val_accuracy, val_loss = validate_epoch()
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


def voc_segmentation(test_model=False):
    def train_epoch():
        min_loss = float('inf')
        max_loss = float('-inf')
        total_loss = 0
        dataset_size = len(train_loader.dataset)
        dataset_batches = len(train_loader)

        tqdm.write(f"\nEpoch {epoch}: ")
        tqdm.write(f" - learning rate: {optimizer.param_groups[0]['lr']}")
        datas = buffer_dataloader(train_loader)

        model.train()
        with tqdm(
                total=dataset_size,
                unit='image') as pbar:
            for (i, (images, labels)) in datas:
                images, labels = images.to(device), labels.unsqueeze(1).float().to(device)

                with torch.cuda.amp.autocast():
                    predict = model(images)
                    loss = criterion(predict, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 记录损失最小值和最大值以及总损失
                min_loss = min(min_loss, loss.item())
                max_loss = max(max_loss, loss.item())
                total_loss += loss.item()

                # 更新进度条
                pbar.set_postfix(loss=loss.item())
                pbar.update(batch_size)

        average_loss = total_loss / dataset_batches
        tqdm.write(
            "\n"
            f"Epoch {epoch} training finished. "
            f"Min loss: {min_loss:.6f}, "
            f"Max loss: {max_loss:.6f}, "
            f"Avg loss: {average_loss:.6f}")
        return average_loss

    def validate_epoch():
        total_loss = 0.0
        num_correct = 0
        num_pixels = 0
        dice_score = 0.0
        dataset_size = len(val_loader.dataset)
        dataset_batches = len(val_loader)

        datas = buffer_dataloader(val_loader)

        model.eval()
        with (tqdm(
                total=dataset_size,
                unit='image') as pbar, torch.no_grad()):
            for (_, (images, labels)) in datas:
                images, labels = images.to(device), labels.unsqueeze(1).float().to(device)

                predict = model(images)

                # 计算损失
                loss = criterion(predict, labels)
                total_loss += loss.item()

                # 预测和计算准确率和 Dice 系数
                predict = torch.sigmoid(predict)
                predict = (predict > 0.5).float()
                num_correct += (predict == labels).sum()
                num_pixels += torch.numel(predict)
                dice_score += ((2 * (predict * labels).sum())
                               / (2 * (predict * labels).sum()
                               + ((predict * labels) < 1).sum()))
                if test_model:
                    save.tensor_to_image(predict, file_name="pred")

                # 更新进度条
                pbar.set_postfix(loss=loss.item())
                pbar.update(batch_size)

        accuracy = num_correct / num_pixels
        average_loss = total_loss / dataset_batches
        dice = dice_score / dataset_batches
        tqdm.write(
            "\n"
            f"Accuracy: {num_correct}/{num_pixels}({accuracy * 100:.2f})%, "
            f"Average loss: {average_loss:.4f}, "
            f"Dice Score: {dice:.2f}")

        return accuracy, dice

    # 超参数设置
    batch_size = 8
    num_workers = 4
    num_epochs = 2
    learning_rate = 1e-8
    weight_decay = 0.0001
    momentum = 0.9
    scheduler_step_size = 2
    scheduler_gamma = 0.5
    image_size = (160, 240)

    # 部署 GPU 设备
    device = assert_on_cuda()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    label_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])

    # 加载数据集
    if not test_model:
        train_dataset = VOCSegmentationDataset(image_transform=image_transform, label_transform=label_transform)
        train_dataset = Subset(train_dataset, range(0, 50))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, persistent_workers=True, pin_memory=True)
    val_dataset = VOCSegmentationDataset(image_transform=image_transform, label_transform=label_transform)
    val_dataset = Subset(val_dataset, range(0, 50))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True, pin_memory=True)

    # 定义模型
    model = UNet_custom_light(num_classes=1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if not test_model:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        scaler = GradScaler()

        # 训练模型
        for epoch in range(num_epochs):
            train_loss = train_epoch()
            val_accuracy, val_loss = validate_epoch()
            scheduler.step()
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f"models/{model.__class__.__name__}.pt")
    else:
        # 加载模型
        model.load_state_dict(torch.load(f"models/{model.__class__.__name__}.pt"))

        # 测试模型
        val_accuracy, val_loss = validate_epoch()
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


if __name__ == '__main__':
    carvana(test_model=False)





