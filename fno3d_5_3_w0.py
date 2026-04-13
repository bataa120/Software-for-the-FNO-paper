#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from Adam import Adam
from scipy.io import savemat

import time

import torch.distributed as dist

from utils import setup_for_distributed, save_on_master, is_main_process


torch.manual_seed(0)
np.random.seed(0)

################################################################
# 3D Fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes5):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes5 = modes5

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(2, in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, self.modes5)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(2, in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, self.modes5)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(2, in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, self.modes5)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(2, in_channels, out_channels,
                                    self.modes1, self.modes2, self.modes3, self.modes5)
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, d, x, y, t), (in_channel, out_channel, d, x, y, t)
        # -> (batch, out_channel, d, x, y, t)
        return torch.einsum("bidxyt,iodxyt->bodxyt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-4), x.size(-3), x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        o_r = torch.zeros(batchsize, self.out_channels,
                          self.modes1, self.modes2, self.modes3, self.modes5,
                          device=x.device)
        o_i = torch.zeros(batchsize, self.out_channels,
                          self.modes1, self.modes2, self.modes3, self.modes5,
                          device=x.device)

        o_r = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes5].real,
            self.weights1[0]
        ) - self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes5].imag,
            self.weights1[1]
        )
        o_i = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes5].real,
            self.weights1[1]
        ) + self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes5].imag,
            self.weights1[0]
        )

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes5] = \
            torch.view_as_complex(torch.stack([o_r, o_i], dim=-1))

        o_r = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes5].real,
            self.weights2[0]
        ) - self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes5].imag,
            self.weights2[1]
        )
        o_i = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes5].real,
            self.weights2[1]
        ) + self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes5].imag,
            self.weights2[0]
        )

        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes5] = \
            torch.view_as_complex(torch.stack([o_r, o_i], dim=-1))

        o_r = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes5].real,
            self.weights3[0]
        ) - self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes5].imag,
            self.weights3[1]
        )
        o_i = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes5].real,
            self.weights3[1]
        ) + self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes5].imag,
            self.weights3[0]
        )

        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes5] = \
            torch.view_as_complex(torch.stack([o_r, o_i], dim=-1))

        o_r = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes5].real,
            self.weights4[0]
        ) - self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes5].imag,
            self.weights4[1]
        )
        o_i = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes5].real,
            self.weights4[1]
        ) + self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes5].imag,
            self.weights4[0]
        )

        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes5] = \
            torch.view_as_complex(torch.stack([o_r, o_i], dim=-1))

        # Return to physical space
        x = torch.fft.irfftn(out_ft, dim=[-3, -2, -1])
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes5, width):
        super(FNO3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes5 = modes5

        self.width = width
        self.padding = 4  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(18, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes5)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes5)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes5)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes5)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(5, 3)

    def forward(self, x):
        batchsize = x.shape[0]
        size_d, size_x, size_y, size_t = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)

        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        size_t = size_t + self.padding
        size_x = size_x + self.padding
        size_y = size_y + self.padding

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_d, size_x, size_y, size_t)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_d, size_x, size_y, size_t)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_d, size_x, size_y, size_t)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_d, size_x, size_y, size_t)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding, :-self.padding]

        x = x.permute(0, 1, 3, 4, 5, 2)

        x = self.fc3(x)
        x = F.gelu(x)

        x = x.permute(0, 5, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


###############################################################################

def create_data_loader():

    #TRAIN_PATH = 'data/uvzu10v10t1t100_115_s5_t25_p.mat'
    #TEST_PATH = 'data/uvzu10v10t1t100_115_s5_t25_p.mat'

    TRAIN_PATH = 'data/uvzu10v10t1t100_115_s10_t25_p_test.mat'
    TEST_PATH  = 'data/uvzu10v10t1t100_115_s10_t25_p_test.mat'

    #ntrain = 1980
    #ntest = 200
    #batch_size = 20

    ntrain = 8
    ntest = 2
    batch_size = 2

    t1 = default_timer()

    ################################################################
    # load data
    ################################################################

    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('au')[:ntrain, [0, 1, 2, 5, 6], :S2, :S3, :T_in]
    train_u = reader.read_field('au')[:ntrain, [0, 1, 2], :S2, :S3, T_in:T+T_in]

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field('au')[-ntest:, [0, 1, 2, 5, 6], :S2, :S3, :T_in]
    test_u = reader.read_field('au')[-ntest:, [0, 1, 2], :S2, :S3, T_in:T+T_in]

    train_a = train_a.reshape(ntrain, S1, S2, S3, T_in)
    train_u = train_u.reshape(ntrain, S11, S2, S3, T)

    test_a = test_a.reshape(ntest, S1, S2, S3, T_in)
    test_u = test_u.reshape(ntest, S11, S2, S3, T)

    print(train_u.shape)
    print(test_u.shape)

    assert S11 == train_u.shape[-4]
    assert S2 == train_u.shape[-3]
    assert S3 == train_u.shape[-2]
    assert T == train_u.shape[-1]
    assert T_in == train_a.shape[-1]

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    train_a = train_a.reshape(ntrain, S1, S2, S3, 1, T_in).repeat([1, 1, 1, 1, T, 1])
    test_a = test_a.reshape(ntest, S1, S2, S3, 1, T_in).repeat([1, 1, 1, 1, T, 1])

    # pad locations (x,y,t)
    gridx = torch.tensor(np.linspace(0, 1, S2), dtype=torch.float)
    gridx = gridx.reshape(1, 1, S2, 1, 1, 1).repeat([1, S1, 1, S3, T, 1])

    gridy = torch.tensor(np.linspace(0, 1, S3), dtype=torch.float)
    gridy = gridy.reshape(1, 1, 1, S3, 1, 1).repeat([1, S1, S2, 1, T, 1])

    gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, 1, T, 1).repeat([1, S1, S2, S3, 1, 1])

    train_a = torch.cat((
        gridx.repeat([ntrain, 1, 1, 1, 1, 1]),
        gridy.repeat([ntrain, 1, 1, 1, 1, 1]),
        gridt.repeat([ntrain, 1, 1, 1, 1, 1]),
        train_a
    ), dim=-1)

    test_a = torch.cat((
        gridx.repeat([ntest, 1, 1, 1, 1, 1]),
        gridy.repeat([ntest, 1, 1, 1, 1, 1]),
        gridt.repeat([ntest, 1, 1, 1, 1, 1]),
        test_a
    ), dim=-1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=batch_size,
        shuffle=False
    )

    t2 = default_timer()
    print('preprocessing finished, time used:', t2 - t1)

    return train_loader, test_loader, y_normalizer


def train(model, train_loader, test_loader, y_normalizer):

    print("Start training...")

    pn = 0

    scheduler_step = 100
    scheduler_gamma = 0.5
    learning_rate = 0.001
    epochs = 500

    batch_size = train_loader.batch_size
    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()

    min_err = float('inf')
    best_epoch = -1
    best_ckpt_path = None

    os.makedirs('model', exist_ok=True)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x).view(batch_size, S11, S2, S3, T)

            mse = F.mse_loss(out, y, reduction='mean')

            y_dec = y_normalizer.decode(y)
            out_dec = y_normalizer.decode(out)
            l2 = myloss(out_dec.view(batch_size, -1), y_dec.view(batch_size, -1))
            l2.backward()

            optimizer.step()

            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).view(batch_size, S11, S2, S3, T)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2 - t1, train_mse, train_l2, test_l2)

        if test_l2 < min_err:
            min_err = test_l2
            best_epoch = ep
            pn = 0

            # deepcopy is important, otherwise best model keeps changing
            best_model = copy.deepcopy(model)

            best_ckpt_path = f'model/ejs_3d_5_3_w{best_epoch}.pth'
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_l2': min_err,
                'modes1': modes1,
                'modes2': modes2,
                'modes3': modes3,
                'modes5': modes5,
                'width': width,
                'S1': S1,
                'S11': S11,
                'S2': S2,
                'S3': S3,
                'T_in': T_in,
                'T': T
            }, best_ckpt_path)

            print('Best epoch =', best_epoch)
            print('Saved checkpoint to', best_ckpt_path)
        else:
            pn += 1

        if pn >= 10:
            break

    print('Finished Training')

    print("Start testing...")

    # use saved best model in memory
    model = best_model
    model.eval()

    test_l2 = 0.0
    index = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).view(batch_size, S11, S2, S3, T)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

            if index == 0:
                print(index)
                x2 = x
                o2 = out
                y2 = y
            else:
                x2 = torch.cat((x2, x), dim=0)
                o2 = torch.cat((o2, out), dim=0)
                y2 = torch.cat((y2, y), dim=0)

            index += 1

    test_l2 /= ntest

    t2 = default_timer()
    print(index, test_l2)

    x1 = x2.detach().cpu().numpy()
    x1 = x1[..., 0, :]
    o1 = o2.detach().cpu().numpy()
    y1 = y2.detach().cpu().numpy()

    savemat('data/out_y_3d_5_3_w.mat', {"x": x1, "out": o1, "y": y1})

    print('Finished Testing')
    print('Best checkpoint path:', best_ckpt_path)


start = time.time()

modes1 = 5
modes2 = 16
modes3 = 32
modes5 = 5

width = 20

S1 = 5
S11 = 3
S2 = 67
S3 = 115

T_in = 15
T = 10

model = FNO3d(modes1, modes2, modes3, modes5, width).cuda()

print(count_params(model))

start_train = time.time()

train_loader, test_loader, y_normalizer = create_data_loader()

train(model, train_loader, test_loader, y_normalizer)

end_train = time.time()
end = time.time()

seconds = end - start
seconds_train = end_train - start_train
print(f"Total elapsed time: {seconds:.2f} seconds, Train stage elapsed time: {seconds_train:.2f} seconds")