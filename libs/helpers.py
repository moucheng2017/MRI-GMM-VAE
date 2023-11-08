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
import nibabel as nib


def direction_average(img, grad):
    # find unique shells - all parameters except gradient directions are the same
    unique_shells = np.unique(grad[:, 3:], axis=0)

    # preallocate
    da_img = np.zeros(img.shape[0:3] + (unique_shells.shape[0],))
    da_grad = np.zeros((unique_shells.shape[0], grad.shape[1]))

    for shell, i in zip(unique_shells, range(0, unique_shells.shape[0])):
        # indices of grad file for this shell
        shell_index = np.all(grad[:, 3:] == shell, axis=1)
        # calculate the spherical mean of this shell - average along final axis
        da_img[..., i] = np.mean(img[..., shell_index], axis=img.ndim - 1)
        # fill in this row of the direction-averaged grad file
        da_grad[i, 3:] = shell

    return da_img, da_grad


def get_data(datadir):
    # get b values and b vectors:
    bvals = np.loadtxt(os.path.join(datadir, 'bvals'))
    bvecs = np.loadtxt(os.path.join(datadir, 'bvecs'))
    bvals = bvals * 1e-03
    bvecs = np.transpose(bvecs)
    # get images and masks:
    img = nib.load(os.path.join(datadir, 'data.nii.gz'))
    mask = nib.load(os.path.join(datadir, 'nodif_brain_mask.nii.gz'))
    img = img.get_data()
    mask = mask.get_data()
    return {'bvals': bvals,
            'bvecs': bvecs,
            'img': img,
            'mask': mask}


def get_grad(bvecs, bvals):
    return np.concatenate((bvecs, bvals[:,None]), axis=1)


def img_infor(img):
    nvoxtotal = np.prod(np.shape(img)[0:3])
    nvol = np.shape(img)[3]
    return nvoxtotal, nvol


def reshape_img(img, nvoxtotal, nvol):
    return np.reshape(img, (nvoxtotal, nvol))
# (todo): keep the img as template for reshaping back the later after training


def reshape_mask(mask, nvoxtotal):
    return np.reshape(mask, (nvoxtotal))
# (todo): keep the mask as template for reshaping back the later after training


def extract_slice(mask, nvoxtotal, no=70):
    masktmp = np.zeros(np.shape(mask))
    masktmp[:, :, no] = mask[:, :, no]
    mask = masktmp
    # mask in voxel format
    maskvox = reshape_mask(mask, nvoxtotal)
    return maskvox


def mask_img(imgvox, maskvox):
    return imgvox[maskvox == 1]


def norm_img(bvals, masked_img, nvol):
    normvol = np.where(bvals==min(bvals))
    img_norm = masked_img/(np.tile(np.mean(masked_img[:, normvol], axis=2), (1, nvol)))
    return img_norm


def cart2sphere(xyz):
    shape = xyz.shape[:-1]
    mu = np.zeros(np.r_[shape, 2])
    r = np.linalg.norm(xyz, axis=-1)
    mu[..., 0] = np.arccos(xyz[..., 2] / r)  # theta
    mu[..., 1] = np.arctan2(xyz[..., 1], xyz[..., 0])
    mu[r == 0] = 0, 0
    return mu


def sphere2cart(theta, phi):
    n = torch.zeros(3, theta.size(0))

    sintheta = torch.sin(theta)
    print(sintheta)
    print(theta)
    print(n)

    n[0, :] = torch.squeeze(sintheta * torch.cos(phi))
    n[1, :] = torch.squeeze(sintheta * torch.sin(phi))
    n[2, :] = torch.squeeze(torch.cos(theta))







