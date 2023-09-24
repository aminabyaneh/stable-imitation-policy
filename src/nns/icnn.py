#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Code for input convex neural nets. Adopted  and modified version of
https://github.com/locuslab/stable_dynamics/blob/master/models/stabledynamics.py.
"""

import sys
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm.auto import tqdm

sys.path.append(os.pardir)
from utils.log_config import logger
# from learn_gmm_ds import prepare_expert_data


class NormalNN(nn.Module):
    def __init__(self):
        super(NormalNN, self).__init__()
        self.zero = torch.zeros(1, 2, device=torch.device("cuda"), requires_grad=True)

        self.f = nn.Sequential(
            nn.Linear(2, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 1, bias=True)
        )

    def forward(self, x):
        return F.softplus(self.f(x) - self.f(self.zero))




class ICNN(nn.Module):
    """ Input convex neural net.
    """

    def __init__(self, layer_sizes, activation=F.leaky_relu):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1, len(layer_sizes) - 1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])
        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i, b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act(z)

        for W, b, U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)

        return F.linear(x, self.W[-1], self.bias[-1]) + \
            (F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0])


class ReHU(nn.Module):
    """Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)


class MakePSD(nn.Module):
    def __init__(self, f, n, eps=0.0001, d=0.01, device=torch.device("cpu")):
        super().__init__()
        self.f = f
        self.zero = torch.zeros(1, n, device=device, requires_grad=True)
        self.eps = eps
        self.d = d
        self.rehu = ReHU(self.d)

    def forward(self, x):
        smoothed_output = F.relu(self.f(x) - self.f(self.zero))
        quadratic_under = self.eps * torch.norm(x)
        return smoothed_output + quadratic_under


class PosDefICNN(nn.Module):
    def __init__(self, layer_sizes, eps=0.0001, negative_slope=0.05):
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
        self.activation = F.tanh

    def reset_parameters(self):
        # copying from PyTorch Linear
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


def prepare_torch_dataset(trajs: np.ndarray, vels: np.ndarray, batch_size: int):
        """ Convert npy data to tensor dataset.

        Args:
            trajs (np.ndarray): Demonstrated trajectories.
            vels (np.ndarray): Demonstrated velocities.
            batch_size (int): Size of data batches for the loader.
        """

        # convert npy to tensor
        x, y = torch.from_numpy(trajs.astype(np.float32)), torch.from_numpy(vels.astype(np.float32))

        x.requires_grad = True
        y.requires_grad = True

        x = x.to(device=torch.device("cuda"))
        y = y.to(device=torch.device("cuda"))

        # generate a dataloader
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    positions_py, velocities_py = prepare_expert_data("G", 7, dir="../../res/expert_models/",
                                                      noise_level=0.01)
    logger.info(f'Handwriting dataset loaded with [{positions_py.shape}, '
                f'{velocities_py.shape}] samples')

    # fit an ICNN to the data
    data_loader = prepare_torch_dataset(positions_py, velocities_py, batch_size=512)
    lpf_icnn = NormalNN()
    lpf_icnn = PosDefICNN([2, 128, 128, 128, 1])

    device = torch.device('cuda')
    lpf_icnn.to(device)

    # build the optimizer
    optimizer = optim.Adam(lpf_icnn.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # lr scheduler
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                            end_factor=0.001, total_iters=30000)

    for epoch in (par := tqdm(range(50000))):
        trajs_t, vels_t = next(iter(data_loader))
        trajs_t, vels_t = trajs_t.to(device), vels_t.to(device)

        # prediction and loss
        lpf_val = lpf_icnn.forward(trajs_t)
        lpf_grad = torch.autograd.grad(lpf_val.sum(), trajs_t, create_graph=True)[0]

        lpf_grad2 = torch.autograd.grad(lpf_grad.sum(), trajs_t, create_graph=True)[0]
        convexity_penalty = torch.sum(F.relu(-lpf_grad2))

        loss = criterion(-1 * lpf_grad, vels_t)

        # optimization and back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # tracking the learning process
        if epoch % 5 == 0:
            par.set_description(f'MSE > {(loss.data.cpu().detach().numpy()):.4f} | '
                                f'LR > {scheduler.get_last_lr()[0]:.4f} | '
                                f'C > {convexity_penalty:.4f}')

    class ds_c:
        def predict(x):
            x = torch.from_numpy(x.astype(np.float32))
            x.requires_grad = True
            x = x.to(device=torch.device("cuda"))
            res = lpf_icnn.forward(x)
            res = -1 * torch.autograd.grad(res.sum(), x, create_graph=True)[0]
            return res.detach().cpu().numpy()

    class lpf_c:
        def predict(x):
            x = torch.from_numpy(x.astype(np.float32))
            x.requires_grad = True
            x = x.to(device=torch.device("cuda"))
            x = x.reshape(1, 2)
            res = lpf_icnn.forward(x)
            return res.detach().cpu().numpy()

    from utils.plot_tools import plot_ds_stream, plot_contours
    plot_ds_stream(ds_c, positions_py[:1000], space_stretch=0.4)
    plot_contours(lpf_c.predict, positions_py[:1000])





class Coupling(nn.Module):
    """Two fully-connected deep nets ``s`` and ``t``, each with num_layers layers and
    The networks will expand the dimensionality up from 2D to 256D, then ultimately
    push it back down to 2D.
    """
    def __init__(self, input_dim=2, mid_channels=256, num_layers=5):
        super().__init__()
        self.input_dim = input_dim
        self.mid_channels = mid_channels
        self.num_layers = num_layers

        #  scale and translation transforms
        self.s = nn.Sequential(*self._sequential(), nn.Tanh())
        self.t = nn.Sequential(*self._sequential())

    def _sequential(self):
        """Compose sequential layers for s and t networks"""
        input_dim, mid_channels, num_layers = self.input_dim, self.mid_channels, self.num_layers
        sequence = [nn.Linear(input_dim, mid_channels), nn.ReLU()]  # first layer
        for _ in range(num_layers - 2):  # intermediate layers
            sequence.extend([nn.Linear(mid_channels, mid_channels), nn.ReLU()])
        sequence.extend([nn.Linear(mid_channels, input_dim)])  # final layer
        return sequence

    def forward(self, x):
        """outputs of s and t networks"""
        return self.s(x), self.t(x)


class RealNVP(nn.Module):
    """Creates an invertible network with ``num_coupling_layers`` coupling layers
    We model the latent space as a N(0,I) Gaussian, and compute the loss of the
    network as the negloglik in the latent space, minus the log det of the jacobian.
    The network is carefully crafted such that the logdetjac is trivially computed.
    """
    def __init__(self, num_coupling_layers):
        super().__init__()
        self.num_coupling_layers = num_coupling_layers

        # model the latent as a

        self.distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(2),
                                                     covariance_matrix=torch.eye(2))
        self.masks = torch.tensor(
           [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype=torch.float32
        )

        # create num_coupling_layers layers in the RealNVP network
        self.layers_list = [Coupling() for _ in range(num_coupling_layers)]

    def forward(self, x, training=True):
        """Compute the forward or inverse transform
        The direction in which we go (input -> latent vs latent -> input) depends on
        the ``training`` param.
        """
        log_det_inv = 0.
        direction = 1
        if training:
            direction = -1

        # pass through each coupling layer (optionally in reverse)
        for i in range(self.num_coupling_layers)[::direction]:
            mask =  self.masks[i]
            x_masked = x * mask
            reversed_mask = 1. - mask
            s, t = self.layers_list[i](x_masked)
            s = s * reversed_mask
            t = t * reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s))
                + x_masked
            )
            # log det (and its inverse) are easily computed
            log_det_inv = log_det_inv + gate * s.sum(1)
        return x, log_det_inv

    def log_loss(self, x):
        """log loss is the neg loglik minus the logdet"""
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -torch.mean(log_likelihood)

mdl = RealNVP(num_coupling_layers=6)
