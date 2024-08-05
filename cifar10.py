import datetime
import os
import time
import torch

from itertools import islice
from pathlib import Path
from modules.neuron import OnlineLIFNode

from torchinfo import summary
from torchview import draw_graph

from torch.nn import MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from torchtoolbox.transform import Cutout

from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop, RandomHorizontalFlip,
    ToTensor
)



########################################################################


import torch.nn as nn
import torch.nn.functional as F


import sys
from models import spiking_vgg
from modules import neuron, surrogate
import argparse
import math
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.utils.data as data

_seed_ = 2022
import random
random.seed(_seed_)

torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

N_EPOCHS = 300
N_CLS = 10
BS = 128
DATA_DIR = Path('/tmp/data')
LOG_DIR = Path('./logs')

# Optimization settings
LR = 0.1
SGD_MOM = 0.9
T_MAX=300

def main():
    parser = argparse.ArgumentParser(description='Classify CIFAR')
    parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=2., type=float)
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument(
        '-model',
        type = str,
        default = 'online_spiking_vgg11_ws'
    )
    parser.add_argument('-drop_rate', type=float, default=0.0)
    parser.add_argument('-stochdepth_rate', type=float, default=0.0)
    parser.add_argument('-cnf', type=str)
    parser.add_argument('-T_train', default=None, type=int)
    parser.add_argument('-dts_cache', type=str, default='./dts_cache')
    parser.add_argument('-loss_lambda', type=float, default=0.05)
    parser.add_argument('-online_update', action='store_true')

    parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')

    args = parser.parse_args()
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
        shuffle=True,
        num_workers=args.j
    )

    d_te = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=trans_te
    )
    l_te = DataLoader(d_te, batch_size=BS, shuffle=False, num_workers=args.j)

    model_name = args.model
    model = spiking_vgg.__dict__[model_name]
    print(model)
    net = model(
        single_step_neuron = OnlineLIFNode,
        tau = args.tau,
        surrogate_function = surrogate.Sigmoid(alpha=4.),
        track_rate = True,
        c_in = 3,
        neuron_dropout = args.drop_rate,
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
    scaler = None

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(LOG_DIR, f'{args.model}_{args.cnf}_T_{args.T}_T_train_{args.T_train}_sgd_lr_{LR}_')
    out_dir += f'CosALR_{T_MAX}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)

    pt_dir = out_dir + '_pt'
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
        print(f'Mkdir {pt_dir}.')


    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    crit = MSELoss()

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(l_tr))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0
        for frame, label in l_tr:
            batch_idx += 1
            frame = frame.float()
            t_step = args.T

            label = label

            batch_loss = 0
            if not args.online_update:
                optimizer.zero_grad()
            for t in range(t_step):
                if args.online_update:
                    optimizer.zero_grad()
                input_frame = frame
                if True:
                    if t == 0:
                        out_fr = net(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, N_CLS).float()
                        mse_loss = crit(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                    loss.backward()

                    if args.online_update:
                        optimizer.step()

                batch_loss += loss.item()
                train_loss += loss.item() * label.numel()
            if not args.online_update:
                optimizer.step()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
            losses.update(batch_loss, input_frame.size(0))
            top1.update(prec1.item(), input_frame.size(0))
            top5.update(prec5.item(), input_frame.size(0))


            train_samples += label.numel()
            train_acc += (total_fr.argmax(1) == label).float().sum().item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx,
                size=len(l_tr),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
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
        top1 = AverageMeter()
        top5 = AverageMeter()
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
                label = label
                t_step = args.T
                total_loss = 0

                for t in range(t_step):
                    input_frame = frame
                    if t == 0:
                        out_fr = net(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, n_classes).float()
                        mse_loss = crit(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                    total_loss += loss

                test_samples += label.numel()
                test_loss += total_loss.item() * label.numel()
                test_acc += (total_fr.argmax(1) == label).float().sum().item()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
                losses.update(total_loss, input_frame.size(0))
                top1.update(prec1.item(), input_frame.size(0))
                top5.update(prec5.item(), input_frame.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=len(l_te),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
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
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))

        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (N_EPOCHS - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
