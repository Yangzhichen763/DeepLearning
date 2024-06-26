
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from lately.DiffusionModel.interpret import linear_interpret
from lossFunc.WeightedLoss import WeightedL2Loss
from model import GaussianDiffusionSampler, GaussianDiffusionTrainer
from modules import UNet
from optim import GradualWarmupScheduler

from utils.os import get_datas_path
from utils.torch import save, load


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root=get_datas_path("CIFAR10"), train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(t=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        t=modelConfig["T"],
        model=net_model,
        betas=linear_interpret(modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]),
        loss_func=WeightedL2Loss()).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        save.as_pt(net_model, save_path=os.path.join(modelConfig["save_weight_dir"], 'ckpt_' + str(e) + '.pt'))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(t=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        load.from_model(model,
                        device=device,
                        load_path=os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]))
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            t=modelConfig["T"],
            model=model,
            betas=linear_interpret(modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save.tensor_to_image(
            saveNoisy,
            save_path=os.path.join(modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]),
            make_grid=True,
            nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save.tensor_to_image(
            sampledImgs,
            save_path=os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"]),
            make_grid=True,
            nrow=modelConfig["nrow"])


if __name__ == '__main__':
    model_config = {
        "state": "eval",  # or eval
        "epoch": 200,
        "batch_size": 4,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./checkpoint/",
        "test_load_weight": "ckpt_0.pt",
        "sampled_dir": "./sampled_images/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
    }
    if model_config["state"] == "train":
        train(model_config)
    else:
        eval(model_config)
