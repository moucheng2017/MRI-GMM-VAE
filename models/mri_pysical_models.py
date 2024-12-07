import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import torch.nn.functional as F
import math


# The following functions are written by Paddy.

__all__ = [
    'ball_stick',
    't2_adc',
    'msdki',
    'ball',
    'stick',
    'get_model_nparams']


def ball_stick(grad, params):
    # nparams = len(params)
    # need to force this shape?
    # params = torch.zeros([1,nparams])

    # extract the parameters
    f = params[:, 0].unsqueeze(1)
    Dpar = params[:, 1].unsqueeze(1)
    D = params[:, 2].unsqueeze(1)
    mu = params[:, 3:4].unsqueeze(1)

    bvecs = grad[:, 0:2]
    bvals = grad[:, 3]

    S = f * stick(grad, Dpar, mu) + (1 - f) * ball(grad, D)

    return S


def t2_adc(grad, params):
    # (todo): Needed for placenta
    # extract the parameters
    T2 = params[:, 0].unsqueeze(1)
    D = params[:, 1].unsqueeze(1)

    grad = torch.from_numpy(grad).float()
    # bvals = grad[:, 3]
    te = grad[:, 4]
    S = torch.exp(-grad[:, 3] * D) * torch.exp(-(te - torch.min(te)) / T2)

    return S


def msdki(grad, params):
    # D = torch.clamp(params[:, 0].unsqueeze(1), min = 0.01, max = 5)
    # K = torch.clamp(params[:, 1].unsqueeze(1), min= 0.001, max=3)

    if torch.is_tensor(grad) is False:
        grad = torch.from_numpy(grad).float()
    if torch.is_tensor(params) is False:
        params = torch.from_numpy(params).float()

    # D = params[:, 0].unsqueeze(1)
    # K = params[:, 1].unsqueeze(1)

    D = torch.clamp(params[:, 0].unsqueeze(1), min=0.01, max=3)
    K = torch.clamp(params[:, 1].unsqueeze(1), min=0.01, max=2)

    bvals = grad[:, 3]

    S = torch.exp(-bvals * D + (bvals ** 2 * D ** 2 * K / 6))

    return S


def ball(grad, D):
    bvals = grad[:, 3]

    S = torch.exp(-bvals * D)
    return S


def stick(grad, Dpar, mu):
    g = grad[:, 0:2]
    bvals = grad[:, 3]

    n = utils.cart2sphere(mu)

    S = torch.exp(-bvals * Dpar * torch.mm(g, n) ** 2)
    return S


def get_model_nparams(model):
    if model == "ball_stick":
        return 5
    if model == "t2_adc":
        return 2
    if model == "msdki":
        return 2

# def ball_stick(grad, params):
#     # extract the parameters
#     f = params[:, 0].unsqueeze(1)
#     Dpar = params[:, 1].unsqueeze(1)
#     Diso = params[:, 2].unsqueeze(1)
#     theta = params[:, 3].unsqueeze(1)
#     phi = params[:, 4].unsqueeze(1)
#     E = f * stick(grad, Dpar, theta, phi) + (1 - f) * ball(grad, Diso)
#     return E
#
#
# def ball(grad, Diso):
#     bvals = grad[:, 3]
#     E = torch.exp(-bvals * Diso)
#     return E
#
#
# def stick(grad, Dpar, theta, phi):
#     g = grad[:, 0:2]
#     bvals = grad[:, 3]
#     n = sphere2cart(theta, phi)
#     print(np.shape(bvals * Dpar))
#     print(n)
#     E = torch.exp(-bvals * Dpar * torch.mm(g, n) ** 2)
#     return E
#
#

