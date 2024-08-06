from itertools import islice
from pathlib import Path
from models.spiking_vgg import OnlineSpikingVGG, spiking_vgg
from modules.surrogate import Sigmoid

from rich.table import Table
from rich import print as rprint

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

import numpy as np
import torch

########################################################################

import torch.nn.functional as F

_seed_ = 2022
import random
random.seed(_seed_)
torch.manual_seed(_seed_)
np.random.seed(_seed_)

N_EPOCHS = 300
N_CLS = 10
BS = 32
DATA_DIR = Path('/tmp/data')
LOG_DIR = Path('/tmp/logs')

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

    d_tr = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=trans_tr
    )
    l_tr = DataLoader(
        d_tr,
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

    tab = Table('Parameter', 'Value', title = 'Parameters')
    params = [
        ('Batch size', BS),
        ('N training batches', len(l_tr))
    ]
    for n, v in params:
        tab.add_row(n, str(v))
    rprint(tab)

    net = OnlineSpikingVGG(
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
    writer = SummaryWriter(LOG_DIR)
    crit = MSELoss()

    for epoch in range(N_EPOCHS):

        print('== Training %3d/%3d ==' % (epoch, N_EPOCHS))

        net.train()

        train_loss = 0
        train_acc = 0
        n_samples = 0
        for idx, (frame, y) in enumerate(islice(l_tr, 5)):
            frame = frame.float()
            batch_loss = 0
            total_fr = torch.zeros((BS, N_CLS))
            optimizer.zero_grad()
            for t in range(N_T_STEPS):
                out_fr = net(frame, init = (t == 0))
                total_fr += out_fr.clone().detach()
                y_one_hot = F.one_hot(y, N_CLS).float()
                mse_loss = crit(out_fr, y_one_hot)
                loss = ((1 - LOSS_LAMBDA) * F.cross_entropy(out_fr, y) + LOSS_LAMBDA * mse_loss) / N_T_STEPS
                loss.backward()

                batch_loss += loss.item()
                train_loss += loss.item() * BS
            optimizer.step()
            n_samples += BS
            train_acc += (total_fr.argmax(1) == y).float().sum().item()

            print('%4d/%4d, batch loss %.4f' % (idx, len(l_tr), batch_loss))

        train_loss /= n_samples
        train_acc /= n_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        print('== Testing %3d/%3d ==' % (epoch, N_EPOCHS))
        net.eval()

        test_loss = 0
        test_acc = 0
        n_samples = 0
        with torch.no_grad():
            for idx, (frame, label) in enumerate(islice(l_te, 3)):
                frame = frame.float()
                total_loss = 0

                total_fr = torch.zeros((BS, N_CLS))
                for t in range(N_T_STEPS):
                    out_fr = net(frame, init = (t == 0))
                    total_fr += out_fr.clone().detach()

                    label_one_hot = F.one_hot(label, N_CLS).float()
                    mse_loss = crit(out_fr, label_one_hot)
                    loss = ((1 - LOSS_LAMBDA) * F.cross_entropy(out_fr, label) + LOSS_LAMBDA * mse_loss) / N_T_STEPS
                    total_loss += loss

                n_samples += BS
                test_loss += total_loss.item() * BS
                test_acc += (total_fr.argmax(1) == label).float().sum().item()

                print('%4d/%4d, batch loss %.4f' % (idx, len(l_te), total_loss))

        test_loss /= n_samples
        test_acc /= n_samples
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

        print(f'train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}')

if __name__ == '__main__':
    main()
