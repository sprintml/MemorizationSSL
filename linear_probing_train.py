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
    batch_size = 128
    max_device_batch_size = 512
    load_batch_size = min(max_device_batch_size, batch_size)
    base_learning_rate = 1e-3
    weight_decay = 0.05
    warmup_epoch = 10
    total_epoch = 90
    steps_per_update = batch_size // load_batch_size
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                 transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    canary = list(range(0, 40000, 1)) + list(range(45000, 50000, 1))
    target = list(range(0, 45000, 1))
    trainDataset_canary = torch.utils.data.Subset(train_dataset, canary)
    trainDataset_target = torch.utils.data.Subset(train_dataset, target)
    train_loader_canary = torch.utils.data.DataLoader(dataset=trainDataset_canary, batch_size=load_batch_size,
                                                      shuffle=True, num_workers=4)
    train_loader_target = torch.utils.data.DataLoader(dataset=trainDataset_target, batch_size=load_batch_size,
                                                      shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_canary = ViT_Classifier(torch.load('./canary_MAE.pt', map_location='cpu').encoder, num_classes=10).to(device)
    model_target = ViT_Classifier(torch.load('./target_MAE.pt', map_location='cpu').encoder, num_classes=10).to(device)

    loss_fn_target = torch.nn.CrossEntropyLoss()
    acc_fn_target = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim_target = torch.optim.AdamW(model_target.parameters(), lr=base_learning_rate * batch_size / 256,
                                     betas=(0.9, 0.999), weight_decay=weight_decay)
    lr_func_target = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                       0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler_target = torch.optim.lr_scheduler.LambdaLR(optim_target, lr_lambda=lr_func_target, verbose=True)

    loss_fn_canary = torch.nn.CrossEntropyLoss()
    acc_fn_canary = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim_canary = torch.optim.AdamW(model_canary.parameters(), lr=base_learning_rate * batch_size / 256,
                                     betas=(0.9, 0.999), weight_decay=weight_decay)
    lr_func_canary = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                       0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler_canary = torch.optim.lr_scheduler.LambdaLR(optim_canary, lr_lambda=lr_func_canary, verbose=True)

    best_val_acc_target = 0
    step_count_target = 0
    optim_target.zero_grad()
    for e in range(total_epoch):
        model_target.train()
        losses_target = []
        acces_target = []
        for img, label in tqdm(iter(train_loader_target)):
            step_count_target += 1
            img = img.to(device)
            label = label.to(device)
            logits = model_target(img)
            loss = loss_fn_target(logits, label)
            acc = acc_fn_target(logits, label)
            loss.backward()
            if step_count_target % steps_per_update == 0:
                optim_target.step()
                optim_target.zero_grad()
            losses_target.append(loss.item())
            acces_target.append(acc.item())
        lr_scheduler_target.step()
        avg_train_loss = sum(losses_target) / len(losses_target)
        avg_train_acc = sum(acces_target) / len(acces_target)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

    torch.save(model_target, './target_classifier.pt')

    best_val_acc_canary = 0
    step_count_canary = 0
    optim_canary.zero_grad()
    for e in range(total_epoch):
        model_canary.train()
        losses_canary = []
        acces_canary = []
        for img, label in tqdm(iter(train_loader_canary)):
            step_count_canary += 1
            img = img.to(device)
            label = label.to(device)
            logits = model_canary(img)
            loss = loss_fn_canary(logits, label)
            acc = acc_fn_canary(logits, label)
            loss.backward()
            if step_count_canary % steps_per_update == 0:
                optim_canary.step()
                optim_canary.zero_grad()
            losses_canary.append(loss.item())
            acces_canary.append(acc.item())
        lr_scheduler_canary.step()
        avg_train_loss = sum(losses_canary) / len(losses_canary)
        avg_train_acc = sum(acces_canary) / len(acces_canary)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

    torch.save(model_canary, './canary_classifier.pt')

    model_canary.eval()
    with torch.no_grad():
        losses_canary = []
        acces_canary = []
        for img, label in tqdm(iter(val_dataloader)):
            img = img.to(device)
            label = label.to(device)
            logits = model_canary(img)
            loss = loss_fncanary(logits, label)
            acc = acc_fn_canary(logits, label)
            losses_canary.append(loss.item())
            acces_canary.append(acc.item())
        avg_val_loss = sum(losses_canary) / len(losses_canary)
        avg_val_acc = sum(acces_canary) / len(acces_canary)
        print(f'for canary classifier, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')

    model_target.eval()
    with torch.no_grad():
        losses_target = []
        acces_target = []
        for img, label in tqdm(iter(val_dataloader)):
            img = img.to(device)
            label = label.to(device)
            logits = model_target(img)
            loss = loss_fn_target(logits, label)
            acc = acc_fn_target(logits, label)
            losses_target.append(loss.item())
            acces_target.append(acc.item())
        avg_val_loss = sum(losses_target) / len(losses_target)
        avg_val_acc = sum(acces_target) / len(acces_target)
        print(f'for target classifier, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')

