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
from functools import reduce, partial
from timeit import default_timer
from Adam import Adam
from scipy.io import savemat

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################
# Fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, d, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.d, self.modes1, self.modes2,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.d, self.modes1, self.modes2,
                dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, d, x, y), (in_channel, out_channel, d, x, y)
        # -> (batch, out_channel, d, x, y)
        return torch.einsum("bidxy,iodxy->bodxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # FFT over the last two dimensions (x, y)
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, self.d,
            x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, d, d1, modes1, modes2, width):
        super(FNO2d, self).__init__()

        self.d = d
        self.d1 = d1
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 6  # pad if input is non-periodic

        # input features = T_in + 2 grid coordinates
        self.fc0 = nn.Linear(17, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.d, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.d, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.d, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.d, self.modes1, self.modes2)

        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(d, d1)

    def forward(self, x):
        # x: (batch, d, x, y, T_in)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # -> (..., T_in + 2)
        x = self.fc0(x)

        # (batch, width, d, x, y)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]

        # (batch, width, x, y, d)
        x = x.permute(0, 1, 3, 4, 2)
        x = self.fc3(x)
        x = F.gelu(x)

        # (batch, d1, x, y, width)
        x = x.permute(0, 4, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_d, size_x, size_y = shape[0], shape[1], shape[2], shape[3]

        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat(batchsize, size_d, 1, size_y, 1)

        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat(batchsize, size_d, size_x, 1, 1)

        return torch.cat((gridx, gridy), dim=-1)


################################################################
# Configs
################################################################

# TRAIN_PATH = 'data/uvzu10v10t1t100_115_s5_t25_p.mat'
# TEST_PATH  = 'data/uvzu10v10t1t100_115_s5_t25_p.mat'

TRAIN_PATH = 'data/uvzu10v10t1t100_115_s10_t25_p_test.mat'
TEST_PATH  = 'data/uvzu10v10t1t100_115_s10_t25_p_test.mat'


#ntrain = 1980
#ntest = 200

ntrain = 8
ntest = 2

modes1 = 16
modes2 = 32
width = 20

#batch_size = 20
batch_size = 2

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

sub = 1
d = 3
d1 = 3
d2 = d - d1

S1 = 67
S2 = 115
T_in = 15
T = 10
step = 1

os.makedirs('model', exist_ok=True)

################################################################
# Load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('au')[:ntrain, :d, ::sub, ::sub, :T_in]
train_u = reader.read_field('au')[:ntrain, :d, ::sub, ::sub, T_in:T + T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('au')[-ntest:, :d, ::sub, ::sub, :T_in]
test_u = reader.read_field('au')[-ntest:, :d, ::sub, ::sub, T_in:T + T_in]

print(train_u.shape)
print(test_u.shape)

assert S2 == train_u.shape[-2]
assert T == train_u.shape[-1]

train_a = train_a.reshape(ntrain, d, S1, S2, T_in)
test_a  = test_a.reshape(ntest, d, S1, S2, T_in)

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

################################################################
# Training and evaluation
################################################################

model = FNO2d(d, d1, modes1, modes2, width).to(device)

print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=scheduler_step, gamma=scheduler_gamma
)

myloss = LpLoss(size_average=False)
min_err = float('inf')
pn = 0
bmodel = None
bep = -1
best_path = None

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0.0
    train_l2_full = 0.0

    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        bs = xx.shape[0]

        loss = 0.0
        pred = None

        for t in range(0, T, step):
            y = yy[:, :d1, :, :, t:t + step]
            im = model(xx)

            loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred[:, :d1, ...], yy[:, :d1, ...])
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    test_l2_step = 0.0
    test_l2_full = 0.0

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            bs = xx.shape[0]

            loss = 0.0
            pred = None

            for t in range(0, T, step):
                y = yy[:, :d1, :, :, t:t + step]
                im = model(xx)

                loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                # im = torch.cat((im, yy[:, -d2:, :, :, t:t + step]), dim=1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred[:, :d1, ...], yy[:, :d1, ...]).item()

    scheduler.step()
    t2 = default_timer()

    print(
        ep,
        t2 - t1,
        train_l2_step / ntrain / (T / step),
        train_l2_full / ntrain,
        test_l2_step / ntest / (T / step),
        test_l2_full / ntest
    )

    if test_l2_full < min_err:
        min_err = test_l2_full
        bmodel = copy.deepcopy(model)
        bep = ep
        pn = 0

        best_path = f'model/model_{d}_{d1}_{modes1}_{modes2}_{bep}.pth'
        torch.save(
            {
                'epoch': bep,
                'model_state_dict': bmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': min_err,
                'd': d,
                'd1': d1,
                'modes1': modes1,
                'modes2': modes2,
                'width': width,
                'T_in': T_in,
                'T': T,
            },
            best_path
        )
        print('Saved best model to', best_path)
    else:
        pn += 1

    if pn >= 10:
        break

print("Start testing...")

model = bmodel
model.eval()

test_l2_step = 0.0
test_l2_full = 0.0
index = 0

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u),
    batch_size=ntest,
    shuffle=False
)

with torch.no_grad():
    for xx1, yy in test_loader:
        xx1 = xx1.to(device)
        yy = yy.to(device)
        bs = xx1.shape[0]

        loss = 0.0
        xx = xx1[..., -T_in:]
        pred = None

        for t in range(0, T, step):
            y = yy[:, :d1, :, :, t:t + step]
            im = model(xx)

            loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-1)

            # im = torch.cat((im, yy[:, -d2:, :, :, t:t + step]), dim=1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        test_l2_step += loss.item()
        test_l2_full += myloss(pred[:, :d1, ...], yy[:, :d1, ...]).item()

        print(index, test_l2_full)

        if index == 0:
            x2 = xx1[:, :d1, ...]
            o2 = pred[:, :d1, ...]
            y2 = yy[:, :d1, ...]

        index += 1

x1 = x2.detach().cpu().numpy()
o1 = o2.detach().cpu().numpy()
y1 = y2.detach().cpu().numpy()

savemat(
    f'data/out_y_2d{d}_{d1}_{modes1}_{modes2}.mat',
    {"x": x1, "out": o1, "y": y1}
)

print('Finished Testing')
print('Best checkpoint:', best_path)