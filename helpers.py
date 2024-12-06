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
    def __init__(self, mri_dir):
        self.mri_dir = mri_dir
        self.bvals = get_data(mri_dir)['bvals']
        self.bvecs = get_data(mri_dir)['bvecs']
        self.grad = get_data(mri_dir)['grad']

        self.img = get_data(mri_dir)['img']
        self.mask = get_data(mri_dir)['mask']

    def __getitem__(self, index):
        X, nvol = extract_slice(img=self.img,
                          mask=self.mask, 
                          no=index)
        
        # X = img_norm(imgvoxtofit=X,
        #              bvals=self.bvals,
        #              nvol=nvol)
        
        return torch.from_numpy(X.astype(np.float32))

    def __len__(self):
        return np.shape(self.mask)[-1]


def direction_average(img, grad):
    # find unique shells - all parameters except gradient directions are the same
    unique_shells = np.unique(grad[:, 3:], axis=0)

    # preallocate
    da_img = np.zeros(img.shape[0:3] + (unique_shells.shape[0]))
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
    img = nib.load(os.path.join(datadir, 'T1w/Diffusion/data.nii.gz')) # (145, 174, 145, 288) height x slices x width x dirs
    mask = nib.load(os.path.join(datadir, 'T1w/Diffusion/nodif_brain_mask.nii.gz')) # (145, 174, 145) height x slices x width x dirs
    
    return {'bvals': bvals,
            'bvecs': bvecs,
            'grad': grad,
            'img': img,
            'mask': mask}

def extract_slice(img,
                  mask, 
                  no):
    # img: nibabel.niftiImage (145, 174, 145, 288)
    # mask: nibabel.niftiImage (145, 174, 145)
    # no: slice number
    
    img_slice = img.slicer[:, :, no:no+1, :].get_fdata().squeeze() # (145, 174, 288)
    mask_slice = mask.slicer[:, :, no:no+1].get_fdata().squeeze() # (145, 174)
    nvol = img_slice.shape[2]
    mask_slice = np.expand_dims(mask_slice, axis=2)
    masked_img_slice = img_slice * mask_slice

    return masked_img_slice, nvol


# def img_norm(imgvoxtofit,
#              bvals,
#              nvol):
#     # normvol = np.where(bvals == min(bvals))
#     # imgvoxtofitnorm = imgvoxtofit / (np.tile(np.mean(imgvoxtofit[:, :, normvol], axis=2), (1, nvol)))
#     min_bval = np.min(bvals)
#     min_bval_indices = [i for i, bval in enumerate(bvals) if bval == min_bval]
#     b0_vols = imgvoxtofit[:, :, min_bval_indices]
#     b0_mean = np.mean(b0_vols, axis=2)
#     imgvoxtofit = imgvoxtofit / b0_mean

#     return imgvoxtofit

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

