from itertools import islice
from pathlib import Path
from modules.neuron import OnlineLIFNode

from rich.table import Table
from rich import print as rprint

from torch.autograd import Function
from torch.nn import (
    AvgPool2d,
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    Parameter,
    Sequential,
    init
)
from torch.nn.functional import (
    conv2d,
    cross_entropy,
    mse_loss, one_hot
)
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
import random
import torch
import torch.nn.functional as F

_seed_ = 2022
random.seed(_seed_)
torch.manual_seed(_seed_)
np.random.seed(_seed_)

N_EPOCHS = 300
N_CLS = 10
BS = 128
DATA_DIR = Path('/tmp/data')
LOG_DIR = Path('/tmp/logs')

# Simulation settings
N_T_STEPS = 6
TAU = 2.0

# Optimization settings
LR = 0.1
SGD_MOM = 0.9
T_MAX = 300
LOSS_LAMBDA = 0.05

def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

class sigmoid(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        return grad_x, None

class SurrogateFunctionBase(Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def spiking_function(x, alpha):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return self.spiking_function(x, self.alpha)

class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha, spiking):
        super().__init__(alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()

class SequentialModule(Sequential):
    def __init__(self, *args):
        super(SequentialModule, self).__init__(*args)

    def forward(self, x, **kwargs):
        for module in self._modules.values():
            if isinstance(module, OnlineLIFNode) or isinstance(module, WrapedSNNOp):
                x = module(x, **kwargs)
            else:
                x = module(x)
        return x

# For scaled weight standardization (see Section 4.4).
class Scale(Module):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x, **kwargs):
        return x * self.scale

class ScaledWSConv2d(Conv2d):
    def __init__(self, in_channels, out_channels):
        super(ScaledWSConv2d, self).__init__(
            in_channels, out_channels,
            3, 1, 1, 1,
            1, True
        )
        self.gain = Parameter(torch.ones(self.out_channels, 1, 1, 1))

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
        weight = weight * self.gain
        return weight

    def forward(self, x):
        return conv2d(
            x, self.get_weight(),
            self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )

class Replace(Function):
    @staticmethod
    def forward(ctx, z1, z1_r):
        return z1_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)

class WrapedSNNOp(Module):
    def __init__(self, op):
        super(WrapedSNNOp, self).__init__()
        self.op = op

    def forward(self, x, require_wrap = True, **kwargs):
        if require_wrap:
            B = x.shape[0] // 2
            spike = x[:B]
            rate = x[B:]
            with torch.no_grad():
                out = self.op(spike).detach()
            in_for_grad = Replace.apply(spike, rate)
            out_for_grad = self.op(in_for_grad)
            output = Replace.apply(out_for_grad, out)
            return output
        else:
            return self.op(x)

class OnlineSpikingVGG(Module):
    def __init__(self, **kwargs):
        super(OnlineSpikingVGG, self).__init__()
        self.features = self.make_layers(**kwargs)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.classifier = SequentialModule(
            WrapedSNNOp(Linear(512, N_CLS))
        )
        self._initialize_weights()

    def forward(self, x, **kwargs):
        if self.training:
            x = self.features(
                x,
                output_type='spike_rate',
                require_wrap=True,
                **kwargs
            )
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(
                x,
                output_type='spike_rate',
                require_wrap=True,
                **kwargs
            )
        else:
            x = self.features(x, require_wrap=False, **kwargs)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x, require_wrap=False, **kwargs)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, ScaledWSConv2d):
                init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(**kwargs):
        print(kwargs)
        layers = []
        in_channels = 3
        first_conv = True

        # Only one VGG arch
        cfg = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        for v in cfg:
            if v == 'M':
                layers += [AvgPool2d(kernel_size=2, stride=2)]
            elif type(v) == int:
                if first_conv:
                    conv2d = ScaledWSConv2d(in_channels, v)
                    first_conv = False
                else:
                    conv2d = WrapedSNNOp(
                        ScaledWSConv2d(in_channels, v)
                    )
                layers += [conv2d, OnlineLIFNode(**kwargs), Scale(2.74)]
                in_channels = v
            else:
                assert False
        return SequentialModule(*layers)

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
        ('N training batches', len(l_tr)),
    ]
    for n, v in params:
        tab.add_row(n, str(v))
    rprint(tab)

    net = OnlineSpikingVGG(
        tau = TAU,
        surrogate_function = Sigmoid(4, True),
        track_rate = True,
        c_in = 3,
        neuron_dropout = 0.0,
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

    for epoch in range(N_EPOCHS):

        print('== Training %3d/%3d ==' % (epoch, N_EPOCHS))

        net.train()

        train_loss = 0
        train_acc = 0
        n_batches = 3
        for idx, (frame, y) in enumerate(islice(l_tr, n_batches)):
            frame = frame.float()
            batch_loss = 0
            total_fr = torch.zeros((BS, N_CLS))
            optimizer.zero_grad()
            for t in range(N_T_STEPS):
                yh = net(frame, init = (t == 0))
                total_fr += yh.clone().detach()
                y_one_hot = one_hot(y, N_CLS).float()

                loss0 = LOSS_LAMBDA * mse_loss(yh, y_one_hot)
                loss1 = (1 - LOSS_LAMBDA) * cross_entropy(yh, y)
                loss = (loss0 + loss1) / N_T_STEPS
                loss.backward()

                batch_loss += loss.item()
                train_loss += loss.item() * BS
            optimizer.step()
            train_acc += (total_fr.argmax(1) == y).float().sum().item()

            print('%4d/%4d, batch loss %.4f' % (idx, n_batches, batch_loss))

        train_loss /= (n_batches * BS)
        train_acc /= (n_batches * BS)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        print('== Testing %3d/%3d ==' % (epoch, N_EPOCHS))
        net.eval()
        test_loss = 0
        test_acc = 0
        n_batches = 3
        with torch.no_grad():
            for idx, (frame, y) in enumerate(islice(l_te, n_batches)):
                frame = frame.float()
                batch_loss = 0

                total_fr = torch.zeros((BS, N_CLS))
                for t in range(N_T_STEPS):
                    yh = net(frame, init = (t == 0))
                    total_fr += yh.clone().detach()
                    y_one_hot = one_hot(y, N_CLS).float()

                    loss0 = LOSS_LAMBDA * mse_loss(yh, y_one_hot)
                    loss1 = (1 - LOSS_LAMBDA) * cross_entropy(yh, y)
                    loss = (loss0 + loss1) / N_T_STEPS
                    batch_loss += loss

                #n_samples += BS
                test_loss += batch_loss.item() * BS
                test_acc += (total_fr.argmax(1) == y).float().sum().item()

                print('%4d/%4d, batch loss %.4f' % (idx, n_batches, batch_loss))

        test_loss /= (n_batches * BS)
        test_acc /= (n_batches * BS)
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

        print('losses %5.3f/%5.3f acc %5.3f/%5.3f, best acc %5.3f' % (train_loss, test_loss,
                                                                      train_acc, test_acc,
                                                                      max_test_acc))

if __name__ == '__main__':
    main()
