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

# Define a data loader
class MRIDataset(utils.Dataset):
    def __init__(self, mri_dir, transform=None):
        self.mri_dir = mri_dir
        self.bvals = get_data(mri_dir)['bvals']
        self.bvecs = get_data(mri_dir)['bvecs']
        self.grad = get_data(mri_dir)['grad']

        self.img = get_data(mri_dir)['img']
        self.mask = get_data(mri_dir)['mask']

    def __getitem__(self, index):
        X = extract_slice(img=self.img,
                          mask=self.mask, 
                          no=index)
        
        X = img_norm(imgvoxtofit=X,
                     bvals=self.bvals)
        
        return torch.from_numpy(X.astype(np.float32))

    def __len__(self):
        return len(np.shape(self.mask)[2])   


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
    bvals = np.loadtxt(os.path.join(datadir, 'T1w', 'Diffusion', 'bvals'))
    bvecs = np.loadtxt(os.path.join(datadir,'T1w', 'Diffusion', 'bvecs'))
    bvals = bvals * 1e-03 # 288
    bvecs = np.transpose(bvecs) # 288 x 3
    grad = np.concatenate((bvecs, bvals[:,None]), axis=1) # 288 x 4
    # get images and brain masks:
    img = nib.load(os.path.join(datadir, 'T1w/T1w_acpc_dc_restore_1.25.nii.gz'))
    mask = nib.load(os.path.join(datadir, 'T1w/Diffusion/nodif_brain_mask.nii.gz'))
    img = img.get_fdata() # 145 x 174 x 145: height x slices x width
    mask = mask.get_fdata() # 145 x 174 x 145: height x slices x width

    return {'bvals': bvals,
            'bvecs': bvecs,
            'grad': grad,
            'img': img,
            'mask': mask}

def extract_slice(img,
                  mask, 
                  nvoxtotal,
                  no):
    masktmp = np.zeros(np.shape(mask))

    img = img[mask == 1]
    masktmp = np.zeros(np.shape(mask))
    masktmp[:, :, no] = mask[:, :, no]
    mask = masktmp
    maskvox = np.reshape(mask, (nvoxtotal))
    imgvoxtofit = img[maskvox == 1]
    return imgvoxtofit

def img_norm(imgvoxtofit,
             bvals,
             nvol):
    normvol = np.where(bvals == min(bvals))
    imgvoxtofitnorm = imgvoxtofit / (np.tile(np.mean(imgvoxtofit[:, normvol], axis=2), (1, nvol)))
    return imgvoxtofitnorm

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


def add_noise(data, scale=0.02):
    data_real = data + np.random.normal(scale=scale, size=np.shape(data))
    data_imag = np.random.normal(scale=scale, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)
    return data_noisy

