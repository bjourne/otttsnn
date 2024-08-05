import datetime
import time
import torch

from itertools import islice
from pathlib import Path
from modules.surrogate import Sigmoid

from torch.nn import MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

from torchtoolbox.transform import Cutout

from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop, RandomHorizontalFlip,
    ToTensor
)

########################################################################

import torch.nn.functional as F
from models import spiking_vgg
from utils import Bar, Logger, AverageMeter

_seed_ = 2022
import random
random.seed(_seed_)

torch.manual_seed(_seed_)

import numpy as np
np.random.seed(_seed_)

N_EPOCHS = 300
N_CLS = 10
BS = 128
DATA_DIR = Path('/tmp/data')
LOG_DIR = Path('./logs')

# Simulation settings
N_T_STEPS = 6
TAU = 2.0

# Optimization settings
LR = 0.1
SGD_MOM = 0.9
T_MAX=300
LOSS_LAMBDA = 0.05

def main():
    trans_tr = Compose([
        RandomCrop(32, padding=4),
        Cutout(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    trans_te = Compose([
        ToTensor(),
        Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    trainset = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=trans_tr
    )
    l_tr = DataLoader(
        trainset,
        batch_size=BS,
        shuffle=True
    )

    d_te = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=trans_te
    )
    l_te = DataLoader(d_te, batch_size=BS, shuffle=False)

    model = spiking_vgg.__dict__['online_spiking_vgg11_ws']
    print(model)
    net = model(
        tau = TAU,
        surrogate_function = Sigmoid(alpha=4),
        track_rate = True,
        c_in = 3,
        neuron_dropout = 0.0,
        grad_with_rate = True,
        fc_hw = 1,
        v_reset = None
    )
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

    optimizer = SGD(
        net.parameters(),
        lr=LR,
        momentum=SGD_MOM,
    )

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX)

    max_test_acc = 0

    writer = SummaryWriter(LOG_DIR / 'logs', purge_step=0)

    crit = MSELoss()

    for epoch in range(N_EPOCHS):
        net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()

        bar = Bar('Processing', max=len(l_tr))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0
        for frame, label in islice(l_tr, 5):
            batch_idx += 1
            frame = frame.float()
            batch_loss = 0
            optimizer.zero_grad()
            for t in range(N_T_STEPS):
                if t == 0:
                    out_fr = net(frame, init=True)
                    total_fr = out_fr.clone().detach()
                else:
                    out_fr = net(frame)
                    total_fr += out_fr.clone().detach()
                label_one_hot = F.one_hot(label, N_CLS).float()
                mse_loss = crit(out_fr, label_one_hot)
                loss = ((1 - LOSS_LAMBDA) * F.cross_entropy(out_fr, label) + LOSS_LAMBDA * mse_loss) / N_T_STEPS
                loss.backward()

                batch_loss += loss.item()
                train_loss += loss.item() * label.numel()
            optimizer.step()

            losses.update(batch_loss, frame.size(0))

            train_samples += label.numel()
            train_acc += (total_fr.argmax(1) == label).float().sum().item()

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            end = time.time()

            bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                batch=batch_idx,
                size=len(l_tr),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
            )
            bar.next()
        bar.finish()

        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(l_te))

        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0
        with torch.no_grad():
            for frame, label in l_te:
                batch_idx += 1
                frame = frame.float()
                total_loss = 0

                for t in range(N_T_STEPS):
                    input_frame = frame
                    if t == 0:
                        out_fr = net(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                    label_one_hot = F.one_hot(label, N_CLS).float()
                    mse_loss = crit(out_fr, label_one_hot)
                    loss = ((1 - LOSS_LAMBDA) * F.cross_entropy(out_fr, label) + LOSS_LAMBDA * mse_loss) / N_T_STEPS
                    total_loss += loss

                test_samples += label.numel()
                test_loss += total_loss.item() * label.numel()
                test_acc += (total_fr.argmax(1) == label).float().sum().item()

                losses.update(total_loss, input_frame.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx,
                    size=len(l_te),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                )
                bar.next()
        bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, LOG_DIR / 'checkpoint_max.pth')

        torch.save(checkpoint, LOG_DIR / 'checkpoint_latest.pth')

        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (N_EPOCHS - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
