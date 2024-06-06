import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lately.UNet.model import UNet_custom_light
from utils.pytorch import assert_on_cuda, save
from utils.general import Trainer, Validator
from utils.pytorch.segment.datasets import BinarySegmentationDataset


def rebarsegmentation(test_model=False):
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
                save.tensor_to_image(predict.detach(), file_name="pred", batch=i)

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
        train_dataset = BinarySegmentationDataset(
            dir_name="RebarSegmentation",
            transform=train_transform,
            mode="train")
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, persistent_workers=True, pin_memory=True)
    val_dataset = BinarySegmentationDataset(
        dir_name="RebarSegmentation",
        transform=val_transform,
        mode="valid")
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


if __name__ == '__main__':
    rebarsegmentation()
