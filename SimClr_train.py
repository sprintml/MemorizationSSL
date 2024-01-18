import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    # canary
    canary = list(range(0, 45000, 1))
    train_data_canary = torch.utils.data.Subset(train_data, canary)
    train_loader_canary = DataLoader(train_data_canary, batch_size=batch_size, shuffle=True, num_workers=2,
                                     pin_memory=True, drop_last=True)
    memory_data_canary = torch.utils.data.Subset(memory_data, canary)
    memory_loader_canary = DataLoader(memory_data_canary, batch_size=batch_size, shuffle=False, num_workers=2,
                                      pin_memory=True)
    test_data_canary = torch.utils.data.Subset(test_data, canary)
    test_loader_canary = DataLoader(test_data_canary, batch_size=batch_size, shuffle=False, num_workers=2,
                                    pin_memory=True)

    # model setup and optimizer config
    model_canary = Model(feature_dim).cuda()
    flops_canary, params_canary = profile(model_canary, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops_canary, params_canary = clever_format([flops_canary, params_canary])
    print('# Model Params: {} FLOPs: {}'.format(params_canary, flops_canary))
    optimizer_canary = optim.Adam(model_canary.parameters(), lr=1e-3, weight_decay=1e-6)
    c = 9

    # training loop
    results_canary = {'train_loss_canary': [], 'test_acc@1_canary': [], 'test_acc@5_canary': []}
    save_name_pre_canary = '{}_{}_{}_{}_{}_canary'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results_canary'):
        os.mkdir('results_canary')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model_canary, train_loader_canary, optimizer_canary)
        results_canary['train_loss_canary'].append(train_loss)
        test_acc_1, test_acc_5 = test(model_canary, memory_loader_canary, test_loader_canary)
        results_canary['test_acc@1_canary'].append(test_acc_1)
        results_canary['test_acc@5_canary'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results_canary, index=range(1, epoch + 1))
        data_frame.to_csv('results_canary/{}_statistics.csv'.format(save_name_pre_canary), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model_canary.state_dict(), 'results_canary/{}_model_canary.pth'.format(save_name_pre_canary))

            # target
            target = list(range(0, 40000, 1)) + list(range(45000, 50000, 1))
            train_data_target = torch.utils.data.Subset(train_data, target)
            train_loader_target = DataLoader(train_data_target, batch_size=batch_size, shuffle=True, num_workers=2,
                                             pin_memory=True, drop_last=True)
            memory_data_target = torch.utils.data.Subset(memory_data, target)
            memory_loader_target = DataLoader(memory_data_target, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)
            test_data_target = torch.utils.data.Subset(test_data, target)
            test_loader_target = DataLoader(test_data_target, batch_size=batch_size, shuffle=False, num_workers=2,
                                            pin_memory=True)

            # model setup and optimizer config
            model_target = Model(feature_dim).cuda()
            flops_target, params_target = profile(model_target, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
            flops_target, params_target = clever_format([flops_target, params_target])
            print('# Model Params: {} FLOPs: {}'.format(params_target, flops_target))
            optimizer_target = optim.Adam(model_target.parameters(), lr=1e-3, weight_decay=1e-6)
            c = 9

            # training loop
            results_target = {'train_loss_target': [], 'test_acc@1_target': [], 'test_acc@5_target': []}
            save_name_pre_target = '{}_{}_{}_{}_{}_target'.format(feature_dim, temperature, k, batch_size, epochs)
            if not os.path.exists('results_target'):
                os.mkdir('results_target')
            best_acc = 0.0
            for epoch in range(1, epochs + 1):
                train_loss = train(model_target, train_loader_target, optimizer_target)
                results_target['train_loss_target'].append(train_loss)
                test_acc_1, test_acc_5 = test(model_target, memory_loader_target, test_loader_target)
                results_target['test_acc@1_target'].append(test_acc_1)
                results_target['test_acc@5_target'].append(test_acc_5)
                # save statistics
                data_frame = pd.DataFrame(data=results_target, index=range(1, epoch + 1))
                data_frame.to_csv('results_target/{}_statistics.csv'.format(save_name_pre_target), index_label='epoch')
                if test_acc_1 > best_acc:
                    best_acc = test_acc_1
                    torch.save(model_target.state_dict(),
                               'results__target/{}_model_target.pth'.format(save_name_pre_target))
