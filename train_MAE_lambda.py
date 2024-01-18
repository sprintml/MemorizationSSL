import os
import argparse
import math
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from tqdm import tqdm

from model import *
from utils import setup_seed

upperbound = 0.25
lowerbound = -0.25




if __name__ == '__main__':
    s = 1
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    to_pil = transforms.ToPILImage()
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])
    setup_seed(42)
    batch_size = 2048
    max_device_batch_size = 512
    load_batch_size = min(max_device_batch_size, batch_size)
    base_learning_rate = 1.5e-4
    weight_decay = 0.05
    mask_ratio = 0.75
    p_lambda = 0.1
    warmup_epoch = 120
    total_epoch = 800
    steps_per_update = batch_size // load_batch_size
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
    pi = torch.tensor(3.1415926).to(device)
    step_count_shadow = 0
    optim_canary.zero_grad()
    model.train()
    encoder_canary = model_canary.encoder.to(device)
    encoder_canary.shuffle = PatchShuffle(0.75)
    losses = []
    for e in range(1):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_loader_canary)):
            step_count_shadow += 1

            img = img.to(device)

            rep, backward_indices = encoder_canary(img)
            predicted_img, mask = model_canary(img)
            rep1, backward_indices1 = encoder_canary(img)
            rep2, backward_indices2 = encoder_canary(img)
            lossMSE = torch.mean((predicted_img - img) ** 2 * mask) / mask_ratio
            lossDIS = torch.mean(
                torch.asin(torch.cosine_similarity(rep.transpose(0, 1).reshape(load_batch_size, -1),
                                                   rep1.transpose(0, 1).reshape(load_batch_size, -1), 0)) / pi)
            if e >= 200:
                loss = (1 - p_lambda) * lossMSE - p_lambda * lossDIS
            else:
                loss = lossMSE
            loss.backward()
            if step_count_shadow % steps_per_update == 0:
                optim_canary.step()
                optim_canary.zero_grad()
            losses.append(loss.item())
        lr_scheduler_canary.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average training loss is {avg_loss}.')
    torch.save(model_canary, f'./my_MAE_canary_lambda_{p_lambda}.pt')

    model_target  = MAE_ViT(mask_ratio=mask_ratio).to(device)
    optim_target  = torch.optim.AdamW(model_canary.parameters(), lr=base_learning_rate * batch_size / 256,
                                     betas=(0.9, 0.95),
                                     weight_decay=weight_decay)
    lr_func_target = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                       0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler_target = torch.optim.lr_scheduler.LambdaLR(optim_target, lr_lambda=lr_func_target, verbose=True)
    pi = torch.tensor(3.1415926).to(device)
    step_count_shadow = 0
    optim_target.zero_grad()
    model_target.train()
    encoder_target = model_canary.encoder.to(device)
    encoder_target.shuffle = PatchShuffle(0.75)
    losses = []
    for e in range(1):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_loader_target)):
            step_count_shadow += 1

            img = img.to(device)

            rep, backward_indices = encoder_target(img)
            predicted_img, mask = model_target(img)
            rep1, backward_indices1 = encoder_target(img)
            rep2, backward_indices2 = encoder_target(img)
            lossMSE = torch.mean((predicted_img - img) ** 2 * mask) / mask_ratio
            lossDIS = torch.mean(
                torch.asin(torch.cosine_similarity(rep.transpose(0, 1).reshape(load_batch_size, -1),
                                                   rep1.transpose(0, 1).reshape(load_batch_size, -1), 0)) / pi)
            if e >= 200:
                loss = (1 - p_lambda) * lossMSE - p_lambda * lossDIS
            else:
                loss = lossMSE
            loss.backward()
            if step_count_shadow % steps_per_update == 0:
                optim_target.step()
                optim_target.zero_grad()
            losses.append(loss.item())
        lr_scheduler_target.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average training loss is {avg_loss}.')
    torch.save(model_canary, f'./my_MAE_target_lambda_{p_lambda}.pt')
