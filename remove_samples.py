import os
import sys
import argparse
import scipy
import torchvision as torchvision
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt
import torch
import numpy as np
import pylab
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from model import *

fw = np.arange(256)
np.random.shuffle(fw)
bw = np.argsort(fw)


def chng(a):
    back_ten = torch.Tensor(np.ones((a.shape[0], 3, 32, 32), dtype='float'))
    for r in range(a.shape[0]):
        for i in range(3):
            for j in range(32):
                for k in range(32):
                    numb_0 = 1024 * i + 32 * j + k
                    di_0 = numb_0 // 96
                    re_0 = numb_0 % 96
                    di_1 = re_0 // 3
                    re_1 = re_0 % 3
                    back_ten[r, i, j, k] = a[r, re_1, di_0, di_1]

    return back_ten


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name,
                                              label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], torch.tensor([int(self.train_labels[index])])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder, data_name, label_name):
    with open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=0)

    with open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=0).reshape(len(y_train), 32, 32, 3)

    x_out = x_train
    y_out = y_train
    return x_out, y_out


def array_gt(vertical: bool):
    index = np.array([])
    if vertical:
        for i in range(8):
            index = np.append(index, np.arange(i, 256 + i, 16))
            index = np.append(index, np.arange((15 - i), 271 - i, 16))
    else:
        for i in range(8):
            index = np.append(index, np.arange(i * 16, (i + 1) * 16, 1))
            index = np.append(index, np.arange((15 - i) * 16, (16 - i) * 16, 1))
    return index


def array_center(center: bool):
    a = np.arange(0, 256, 1)
    a = a.reshape(16, 16)
    b = []
    if center:
        for i in range(7):
            b = np.append(b, a[0 + i, 0 + i:16 - i])
            b = np.append(b, a[15 - i, 0 + i:16 - i])
            b = np.append(b, a[1 + i:15 - i, 0 + i])
            b = np.append(b, a[1 + i:15 - i, 15 - i])
        b = np.append(b, a[7, 7:9])
        b = np.append(b, a[8, 7:9])
    else:
        b = np.append(b, a[7, 7:9])
        b = np.append(b, a[8, 7:9])
        for j in range(7):
            i = 6 - j
            b = np.append(b, a[0 + i, 0 + i:16 - i])
            b = np.append(b, a[15 - i, 0 + i:16 - i])
            b = np.append(b, a[1 + i:15 - i, 0 + i])
            b = np.append(b, a[1 + i:15 - i, 15 - i])
    return b


def array_gt_lr(left: bool):
    a = np.arange(0, 256, 1)
    a = a.reshape(16, 16)
    b = np.array([])
    if left:
        for i in range(16):
            b = np.append(b, a[:, 0 + i])
    else:
        for j in range(16):
            i = 15 - j
            b = np.append(b, a[:, 0 + i])
    return b


def random_indexes(mod: str):
    # forward_indexes = np.arange(size)
    # np.random.shuffle(forward_indexes) #Disable the shuffling?
    # backward_indexes = np.argsort(forward_indexes)
    # disabled shuffling so we use from the same overall set even when we change the ratio
    if mod == 'top':
        forward_indexes = abs(np.sort(-np.arange(256)))
        backward_indexes = np.argsort(forward_indexes)
    elif mod == 'lr':
        forward_indexes = array_gt(vertical=True)
        backward_indexes = np.argsort(forward_indexes)
    elif mod == 'tb':
        forward_indexes = array_gt(vertical=False)
        backward_indexes = np.argsort(forward_indexes)
    elif mod == 'left':
        forward_indexes = array_gt_lr(left=True)
        backward_indexes = np.argsort(forward_indexes)
    elif mod == 'right':
        forward_indexes = array_gt_lr(left=False)
        backward_indexes = np.argsort(forward_indexes)
    elif mod == 'center':
        forward_indexes = array_center(center=True)
        backward_indexes = np.argsort(forward_indexes)
    elif mod == 'around':
        forward_indexes = array_center(center=False)
        backward_indexes = np.argsort(forward_indexes)
    elif mod == 'bottom':
        forward_indexes = np.arange(256)
        backward_indexes = np.argsort(forward_indexes)
    else:
        forward_indexes = np.arange(256)
        np.random.shuffle(forward_indexes)
        backward_indexes = np.argsort(forward_indexes)

    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio: float, mod: str) -> None:
        super().__init__()
        self.ratio = ratio
        self.mod = mod

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(self.mod) for _ in range(B)]
        # print("indexes0", indexes[0])
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        # print("forward_indexes", forward_indexes.shape) 256 x 1
        # print("backward_indexes", backward_indexes.shape) 256 x 1
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    datapath = './data'
    datapath1 = './data/cifar-10-batches-py-modi'
    # dataset = torchvision.datasets.LFWPeople(root=datapath, download=True, split='test',
    # transform=Compose([Resize(32), ToTensor(), Normalize(0.5, 0.5)]))
    CIFAR_10_Dataset = torchvision.datasets.CIFAR10(datapath, train=True, download=False,
                                                    transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    sublist = list(range(0, 25000, 1))
    subset = torch.utils.data.Subset(CIFAR_10_Dataset, sublist)
    dataloader_A = torch.utils.data.DataLoader(dataset=subset, batch_size=1,
                                               shuffle=False, num_workers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model2 = torch.load('./my_MAE_canary_cifar.pt', map_location='cpu').to(device).eval()
    model1 = torch.load('./my_MAE_cifar.pt', map_location='cpu').to(device).eval()
    encoder1 = model1.encoder.to(device)
    encoder1.shuffle = PatchShuffle(0, 'rnd')
    encoder2 = model2.encoder.to(device)
    encoder2.shuffle = PatchShuffle(0, 'rnd')
    i = 0
    repout = []
    prob_B = []
    prob_A = []
    l2dist = nn.PairwiseDistance(p=2)
    cosdist = nn.CosineSimilarity(dim=1, eps=1e-6)
    s = 1
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])
    for i, (imgs, _) in enumerate(dataloader_A):
        imgs = imgs.to(device)

        for image in imgs:
            distorigtemp = 0
            fractionorigtemp = 0
            distcanarytemp = 0
            fractioncanarytemp = 0
            dist_B = []
            dist_A = []
            for j in range(5):
                aug_image = to_pil(image)
                aug_image = data_transforms(image)

                rep1_B, backward_indices = encoder1(image.unsqueeze(0).to(device).reshape(1, 3, 32, 32))
                rep1_B = torch.stack(list(rep1_B), dim=-1)
                rep2_B, backward_indices = encoder1(aug_image.unsqueeze(0).to(device).reshape(1, 3, 32, 32))
                rep2_B = torch.stack(list(rep2_B), dim=-1)
                rep1_A, backward_indices = encoder2(image.unsqueeze(0).to(device).reshape(1, 3, 32, 32))
                rep1_A = torch.stack(list(rep1_A), dim=-1)
                rep2_A, backward_indices = encoder2(aug_image.unsqueeze(0).to(device).reshape(1, 3, 32, 32))
                rep2_A = torch.stack(list(rep2_A), dim=-1)

                rep1_B = torch.nn.functional.normalize(rep1_B, dim=1)
                rep2_B = torch.nn.functional.normalize(rep2_B, dim=1)
                rep1_A = torch.nn.functional.normalize(rep1_A, dim=1)
                rep2_A = torch.nn.functional.normalize(rep2_A, dim=1)
                dist_B.append(l2dist(rep1_B.reshape(1, 257 * 192), rep2_B.reshape(1, 257 * 192)).item())
                dist_A.append(l2dist(rep1_A.reshape(1, 257 * 192), rep2_A.reshape(1, 257 * 192)).item())

                fractionorigtemp = np.mean(dist_B)
                fractioncanarytemp = np.mean(dist_A)

        prob_B.append(fractionorigtemp)
        prob_A.append(fractioncanarytemp)

    prob_B = np.array(prob_B)
    prob_B = prob_B / max(prob_B)
    prob_A = np.array(prob_A)
    prob_A = prob_A / max(prob_A)
    diff_prob = prob_B - prob_A
    diff_prob = diff_prob / max(diff_prob)
    index = np.argsort(diff_prob)
    dataset_sort = np.flip(index)
    subset_500 = dataset_sort[0:500]
    subset_1000 = dataset_sort[0:1000]
    subset_2000 = dataset_sort[0:2000]
    subset_4000 = dataset_sort[0:4000]
    subset_8000 = dataset_sort[0:8000]
    subset_16000 = dataset_sort[0:16000]
    scipy.io.savemat('subset_list.mat', {'subset_500': subset_500, 'subset_1000': subset_1000, 'subset_2000':
        subset_2000, 'subset_4000': subset_4000, 'subset_8000': subset_8000, 'subset_16000': subset_16000})
