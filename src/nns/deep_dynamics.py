#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Code for deep dynamic models to train LPF and DS jointly. Adopted  and modified version of
https://github.com/locuslab/stable_dynamics/blob/master/models/stabledynamics.py.
"""

import torch
import torch.nn.functional as F

from torch import nn
from typing import List


class NormalNN(nn.Module):
    """ A conventional feed-forward neural network.
    """

    def __init__(self, layer_sizes: List[int] = [2, 512, 512, 512, 256, 256, 128, 64, 2],
                 activation=nn.LeakyReLU):

        super(NormalNN, self).__init__()
        self.__model = nn.Sequential()

        # Add input layer
        input_size = layer_sizes[0]
        self.__model.add_module("il", nn.Linear(input_size, layer_sizes[1]))
        self.__model.add_module("act_0", activation())

        # Add hidden layers
        for i in range(1, len(layer_sizes) - 2):
            self.__model.add_module(f"hl_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.__model.add_module(f"act_{i}", activation())

        self.__model.add_module(f"ol_{i}", nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):
        return self.__model(x)


class PosDefICNN(nn.Module):
    def __init__(self, layer_sizes, eps, negative_slope):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1, len(layer_sizes) - 1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l))
                                   for l in layer_sizes[1:]])
        self.eps = eps
        self.negative_slope = negative_slope
        self.reset_parameters()
        self.activation = F.leaky_relu

    def reset_parameters(self):
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.activation(z)

        for W, bias, U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W) + F.linear(z, F.softplus(U), bias) * self.negative_slope
            z = self.activation(z)

        z = F.linear(x, self.W[-1]) + F.linear(z, F.softplus(self.U[-1]), self.bias[-1])
        return z + self.eps * (x ** 2).sum(1)[:, None]


class Calibrate(nn.Module):
    def __init__(self, f, n=2, device=torch.device("cpu")):
        super().__init__()
        self.f = f
        self.zero = torch.zeros(1, n, device=device, requires_grad=True)

    def forward(self, x):
        output = self.f(x) - self.f(self.zero)
        return output


class Dynamics(nn.Module):
    def __init__(self, fhat, V, alpha=0.01, relaxed: bool = True):
        super().__init__()
        self.fhat = fhat
        self.V = V
        self.alpha = alpha
        self.relaxed = relaxed

    def forward(self, x):
        fx = self.fhat(x)
        Vx = self.V(x)

        if self.relaxed:
            gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True)[0]
            rv = fx - \
                gV * (F.relu((gV * fx).sum(dim=1)) / (gV ** 2).sum(dim=1))[:,None]
            rv = torch.nan_to_num(rv)

        else:
            gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True)[0]
            rv = fx - \
                    gV * (F.relu((gV * fx).sum(dim=1) + \
                            self.alpha * Vx[:,0]) / (gV ** 2).sum(dim=1))[:,None]
            rv = torch.nan_to_num(rv)
        return rv


def joint_lpf_ds_model(device, lsd=2):

    # dynamics function to learn
    fhat = Calibrate(NormalNN(), device=device)
    fhat.to(device)

    # convex Lyapunov function
    V = Calibrate(PosDefICNN([lsd, 512, 512, 256, 256, 64, 1], eps=0.001, negative_slope=0.01),
                  device=device)
    V.to(device)

    # joint dynamics model
    f = Dynamics(fhat, V, alpha=0.01, relaxed=False)
    f.to(device)

    return f, fhat, V


class SRVDMetric(nn.Module):
    """SRVD Metric for curve similarity.
    """
    def __init__(self):
        super(SRVDMetric, self).__init__()

    def forward(self, vel1, vel2):
        squared_velocity_diff = (vel1 - vel2)**2
        sum_squared_velocity_diff = torch.sum(squared_velocity_diff)
        srvd_distance = torch.sqrt(sum_squared_velocity_diff)
        return srvd_distance