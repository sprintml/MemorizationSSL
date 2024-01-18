import os
import argparse
import math
import torch
import torchvision
from torch.utils.data import Dataset

from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed

if __name__ == '__main__':
    setup_seed(42)
    batch_size = 2048
    max_device_batch_size = 512
    load_batch_size = min(max_device_batch_size, batch_size)
    base_learning_rate = 1.5e-4
    weight_decay = 0.05
    mask_ratio = 0.75
    warmup_epoch = 120
    total_epoch = 800
    steps_per_update = batch_size // load_batch_size
    data_tf = Compose([ToTensor(), Normalize([0.5], [0.5])])
    datapath = './data'
    train_dataset = torchvision.datasets.CIFAR10(datapath, train=True, download=False,
                                                 transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    target = list(range(0, 40000, 1)) + list(range(45000, 50000, 1))
    canary = list(range(0, 45000, 1))
    trainDataset_canary = torch.utils.data.Subset(train_dataset, canary)
    trainDataset_target = torch.utils.data.Subset(train_dataset, target)
    train_loader_canary = torch.utils.data.DataLoader(dataset=trainDataset_canary, batch_size=load_batch_size,
                                                      shuffle=True, num_workers=2)
    train_loader_target = torch.utils.data.DataLoader(dataset=trainDataset_target, batch_size=load_batch_size,
                                                      shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_canary = MAE_ViT(mask_ratio=mask_ratio).to(device)
    optim_canary = torch.optim.AdamW(model_canary.parameters(), lr=base_learning_rate * batch_size / 256,
                                     betas=(0.9, 0.95),
                                     weight_decay=weight_decay)
    lr_func_canary = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                       0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler_canary = torch.optim.lr_scheduler.LambdaLR(optim_canary, lr_lambda=lr_func_canary, verbose=True)
    step_count_canary = 0
    optim_canary.zero_grad()
    for e in range(total_epoch):
        model_canary.train()
        losses = []
        for img, label in tqdm(iter(train_loader_canary)):
            step_count_canary += 1
            img = img.to(device)
            predicted_img, mask = model_canary(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / mask_ratio
            loss.backward()

            if step_count_canary % steps_per_update == 0:
                optim_canary.step()
                optim_canary.zero_grad()
            losses.append(loss.item())
        lr_scheduler_canary.step()
        avg_loss = sum(losses) / len(losses)
        if e % 50 == 0:
            checkpoint(model_shadow, f"./canary_checkpoints/epoch-{e}.pt")
        print(f'In epoch {e}, average training loss is {avg_loss}.')

    torch.save(model_canary, './canary_MAE.pt')

    model_target = MAE_ViT(mask_ratio=mask_ratio).to(device)
    optim_target = torch.optim.AdamW(model_target.parameters(), lr=base_learning_rate * batch_size / 256,
                                     betas=(0.9, 0.95),
                                     weight_decay=weight_decay)
    lr_func_target = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                       0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler_target = torch.optim.lr_scheduler.LambdaLR(optim_target, lr_lambda=lr_func_target, verbose=True)
    step_count_target = 0
    optim_target.zero_grad()
    for e in range(total_epoch):
        model_target.train()
        losses = []
        for img, label in tqdm(iter(train_loader_target)):
            step_count_target += 1
            img = img.to(device)
            predicted_img, mask = model_target(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / mask_ratio
            loss.backward()
            if step_count_target % steps_per_update == 0:
                optim_target.step()
                optim_target.zero_grad()
            losses.append(loss.item())
        lr_scheduler_target.step()
        avg_loss = sum(losses) / len(losses)
        if e % 50 == 0:
            checkpoint(model_shadow, f"./checkpoints/target_epoch-{e}.pt")
        print(f'In epoch {e}, average training loss is {avg_loss}.')

    torch.save(model_target, './target_MAE.pt')

