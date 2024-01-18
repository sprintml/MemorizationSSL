import torch
from torchvision.transforms import ToTensor, Compose, Normalize
import torchvision
from torch.utils.data import DataLoader
from utils import augment, adjust_learning_rate, optim
from model import VICRegNet
from loss import sim_loss, cov_loss, std_loss
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VICReg Training')
parser.add_argument('--path', default='./data', help='path to dataset')
parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
parser.add_argument('--device', default='cuda', type=str, help='device for training')
parser.add_argument('--l', default=25, help='coefficients of the invariance')
parser.add_argument('--mu', default=25, help='coefficients of the variance')
parser.add_argument('--nu', default=1, help='coefficients of the covariance')
parser.add_argument('--weight_decay', default=1e-6, help='weight decay')
parser.add_argument('--lr', default=0.2, help='weight decay')
parser.add_argument('--epoch', default=1000, help='number of epochs')
parser.add_argument('--log_dir', default=r'logs', help='directory to save logs')
parser.add_argument('--save_chpt', default='checkpoints', help='path to save checkpoints')
parser.add_argument('--save_freq', default=1000, help='step frequency to save checkpoints')


def main():
    print('Training Starts')
    args = parser.parse_args()
    os.makedirs(os.path.join('.', args.log_dir, 'canary'), exist_ok=True)
    os.makedirs(os.path.join('.', args.log_dir, 'target'), exist_ok=True)
    os.makedirs(os.path.join('.', args.save_chpt, 'canary'), exist_ok=True)
    os.makedirs(os.path.join('.', args.save_chpt, 'target'), exist_ok=True)
    writer_canary = SummaryWriter(log_dir=os.path.join('.', args.log_dir, 'canary'))
    writer_target = SummaryWriter(log_dir=os.path.join('.', args.log_dir, 'target'))
    train_dataset = torchvision.datasets.CIFAR10(args.path, train=True, download=False,
                                                 transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    target = list(range(0, 40000, 1)) + list(range(45000, 50000, 1))
    canary = list(range(0, 45000, 1))
    traindataset_canary = torch.utils.data.Subset(train_dataset, canary)
    traindataset_target = torch.utils.data.Subset(train_dataset, target)
    train_loader_canary = torch.utils.data.DataLoader(dataset=traindataset_canary, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=2)
    train_loader_target = torch.utils.data.DataLoader(dataset=traindataset_target, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=2)

    model_canary = VICRegNet().to(args.device)
    model_target = VICRegNet().to(args.device)
    optimizer_canary = optim(model_canary, args.weight_decay)
    optimizer_target = optim(model_target, args.weight_decay)

    for epoch in range(0, args.epoch):
        loop = tqdm(enumerate(train_loader_canary, start=epoch * len(train_loader_canary)),
                    total=len(train_loader_canary), leave=False)
        for step, ((img_a, img_b), _) in loop:
            adjust_learning_rate(args, optimizer_canary, train_loader_canary, step)
            optimizer_canary.zero_grad()
            repr_a = model_canary(img_a.to(args.device))
            repr_b = model_canary(img_b.to(args.device))

            _sim_loss_canary = sim_loss(repr_a, repr_b)
            _std_loss_canary = std_loss(repr_a, repr_b)
            _cov_loss_canary = cov_loss(repr_a, repr_b)

            loss_canary = args.l * _sim_loss_canary + args.mu * _std_loss_canary + args.nu * _cov_loss_canary
            loss_canary.backward()
            optimizer_canary.step()
            writer_canary.add_scalar("Loss/train", loss_canary, epoch)

            if step % int(args.save_freq) == 0 and step:
                with open(os.path.join('.', args.log_dir, 'canary', 'logs.txt'), 'a') as log_file:
                    log_file.write(f'Epoch: {epoch}, Step: {step}, Train loss: {loss_canary.cpu().detach().numpy()} \n')

                state = dict(epoch=epoch + 1, model=model_canary.state_dict(),
                             optimizer=optimizer_canary.state_dict())

                torch.save(state, os.path.join('.', args.save_chpt, 'canary', f'checkpoint_{step}_steps.pth'))
            loop.set_description(f'Epoch [{epoch}/{args.epoch}]')
            loop.set_postfix(loss=loss_canary.cpu().detach().numpy())

        print(f'Loss for epoch {epoch} is {loss_canary.cpu().detach().numpy()}')
    print('End of the Training. Saving final checkpoints.')
    state = dict(epoch=args.epoch, model=model_canary.state_dict(),
                 optimizer=optimizer_canary.state_dict())
    torch.save(state, os.path.join('.', args.save_chpt, 'canary', 'final_checkpoint.pth'))
    writer_canary .flush()
    writer_canary .close()

    for epoch in range(0, args.epoch):
        loop = tqdm(enumerate(train_loader_target, start=epoch * len(train_loader_target)),
                    total=len(train_loader_target), leave=False)
        for step, ((img_a, img_b), _) in loop:
            adjust_learning_rate(args, optimizer_target, train_loader_target, step)
            optimizer_target.zero_grad()
            repr_a = model_target(img_a.to(args.device))
            repr_b = model_target(img_b.to(args.device))

            _sim_loss_target = sim_loss(repr_a, repr_b)
            _std_loss_target = std_loss(repr_a, repr_b)
            _cov_loss_target = cov_loss(repr_a, repr_b)

            loss_target = args.l * _sim_loss_target + args.mu * _std_loss_target + args.nu * _cov_loss_target
            loss_target.backward()
            optimizer_target.step()
            writer_target.add_scalar("Loss/train", loss_target, epoch)

            if step % int(args.save_freq) == 0 and step:
                with open(os.path.join('.', args.log_dir, 'target', 'logs.txt'), 'a') as log_file:
                    log_file.write(f'Epoch: {epoch}, Step: {step}, Train loss: {loss_target.cpu().detach().numpy()} \n')

                state = dict(epoch=epoch + 1, model=model_target.state_dict(),
                             optimizer=optimizer_target.state_dict())

                torch.save(state, os.path.join('.', args.save_chpt, 'target', f'checkpoint_{step}_steps.pth'))
            loop.set_description(f'Epoch [{epoch}/{args.epoch}]')
            loop.set_postfix(loss=loss_target.cpu().detach().numpy())

        print(f'Loss for epoch {epoch} is {loss_target.cpu().detach().numpy()}')
    print('End of the Training. Saving final checkpoints.')
    state = dict(epoch=args.epoch, model=model_target.state_dict(),
                 optimizer=optimizer_target.state_dict())
    torch.save(state, os.path.join('.', args.save_chpt, 'target', 'final_checkpoint.pth'))
    writer_target.flush()
    writer_target.close()


if '__main__' == __name__:
    main()
