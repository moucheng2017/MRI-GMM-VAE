import argparse
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
from arguments import get_args
from libs.helpers import get_data, get_grad, img_infor, reshape_img, extract_slice, mask_img, norm_img
from models.base import Net
from models.vae_gau import Net_VAE
import scipy.ndimage as ndimage
from libs.helpers import cart2sphere, sphere2cart
from libs.helpers import direction_average
from pathlib import Path
# ,pred_slice


def pred(args):

    pred_slice = args.slice_index
    # Generates the predicted images using a trained network
    # The code executes in three main sections 1) Loading network 2) Loading Data and 3) Creating the prediction

    ## 2) Load the data
    if args.dataset.name == 'hpc':
        # Load diffusion data:
        directory = args.dataset.data_dir
        bvals = np.loadtxt(directory + '/T1w/Diffusion/bvals')
        bvals *= 1e-03

        bvecs = np.loadtxt(directory + '/T1w/Diffusion/bvecs')
        bvecs = np.transpose(bvecs)

        # grad = np.concatenate((bvecs, bvals[:, None]), axis=1)

        # The following will be put into data loader function later ===================================
        img = nib.load(directory + '/T1w/Diffusion/data.nii.gz')
        mask = nib.load(directory + '/T1w/Diffusion/nodif_brain_mask.nii.gz')
        img = img.get_fdata()
        mask = mask.get_fdata()

    elif args.dataset.name == 'placenta':
        # Load diffusion data:
        directory = args.dataset.data_dir
        grad = np.loadtxt(directory + '/grad_echo_inv_TE4.txt')
        # print(np.shape(grad))
        bvals = grad[:, 3]
        bvals *= 1e-03

        # bvecs = np.loadtxt(directory + '/grad_echo_inv_TE4')
        bvecs = grad[:, 0:2]
        bvecs = np.transpose(bvecs)

        # grad = np.concatenate((bvecs, bvals[:, None]), axis=1)

        # The following will be put into data loader function later ===================================
        img = nib.load(directory + '/chip0244/chip0244_20_20_3401_T2MEdiff_moco_abs.all4e.nii.gz')
        mask = nib.load(directory + '/chip0244/chip0244_20_20_3401_T2MEdiff_moco_abs.e01_placenta_and_uterine_wall_mask_pjs.nii.gz')
        img = img.get_fdata() + 1e-8
        mask = mask.get_fdata()
        # img, grad = direction_average(img, grad)

    else:
        raise NotImplementedError

    nvoxtotal = np.prod(np.shape(img)[0:3])
    nvol = np.shape(img)[3]
    #imgvox = np.reshape(img, (nvoxtotal, nvol)) #My comp ain't happy w/ this coz it's too data demanding

    masktmp = np.zeros(np.shape(mask))
    masktmp[:, :, pred_slice] = mask[:, :, pred_slice]
    mask = masktmp
    maskvox = np.reshape(mask, (nvoxtotal))

    #imgvoxtofit = imgvox[maskvox == 1]
    imgvoxtofit = img[mask==1]

    normvol = np.where(bvals == min(bvals))

    imgvoxtofitnorm = imgvoxtofit / (np.tile(np.mean(imgvoxtofit[:, normvol], axis=2), (1, nvol)))
    # =============================================================================================

    b_values_no0 = torch.FloatTensor(bvals)
    gradient_directions_no0 = torch.FloatTensor(bvecs)

    model_full_path = args.save_path + '/' + args.model_full_path
    net = torch.load(model_full_path) # "C:\\Users\\tobia\\OneDrive - University College London\\Documents\\MICCAI23 Helping Mou and Paddy\\networkSaves\\/models_hpc/gaussian/gaussian_dim_5_par_5_mri_ball_stick_std_1.0_lr_0.001_epoch_400_alpha_0.0001_anneal_1e-05_warm_0.08/gaussian_dim_5_par_5_mri_ball.pt"
    
    ## 3) Predict
    if args.model.mri == 'ball_stick':
        # X_real_pred = X_real_pred.numpy()
        # D_par = D_par.numpy()
        # D_iso = D_iso.numpy()
        # mu_cart = mu_cart.numpy()
        # Fp = Fp.numpy()

        net.eval()
        mc_samples = args.mc_samples

        # fig, ax = plt.subplots(6, mc_samples, figsize=(30, 30))

        for i in range(mc_samples):
            with torch.no_grad():
                if args.model.name == 'mlp':
                    outputs = net(torch.from_numpy(imgvoxtofitnorm.astype(np.float32)))
                    # X_real_pred, D_par, D_iso, mu_cart, Fp = net(torch.from_numpy(imgvoxtofitnorm.astype(np.float32)))
                elif args.model.name == 'gaussian':
                    outputs = net(torch.from_numpy(imgvoxtofitnorm.astype(np.float32)))
                    # X_real_pred, D_par, D_iso, mu_cart, Fp, mu, log_var, _ = net(torch.from_numpy(imgvoxtofitnorm.astype(np.float32)))
                else:
                    raise NotImplementedError

            D_par = outputs['d_par'].numpy()
            D_iso = outputs['d_iso'].numpy()
            mu_cart = outputs['mu_cart'].numpy()
            Fp = outputs['fp'].numpy()

            mu_cart_transposed = mu_cart.transpose()
            mu_vals = cart2sphere(mu_cart_transposed)
            theta = mu_vals[:, 0]
            phi = mu_vals[:, 1]

            D_par_vox = np.zeros(np.shape(maskvox))
            D_par_vox[maskvox == 1] = np.squeeze(D_par[:])
            D_par_map = ndimage.rotate(np.reshape(D_par_vox, np.shape(mask)), 90, reshape=False)

            D_iso_vox = np.zeros(np.shape(maskvox))
            D_iso_vox[maskvox == 1] = np.squeeze(D_iso[:])
            D_iso_map = ndimage.rotate(np.reshape(D_iso_vox, np.shape(mask)), 90, reshape=False)

            theta_vox = np.zeros(np.shape(maskvox))
            theta_vox[maskvox == 1] = np.squeeze(theta[:])
            theta_map = ndimage.rotate(np.reshape(theta_vox, np.shape(mask)), 90, reshape=False)

            phi_vox = np.zeros(np.shape(maskvox))
            phi_vox[maskvox == 1] = np.squeeze(phi[:])
            phi_map = ndimage.rotate(np.reshape(phi_vox, np.shape(mask)), 90, reshape=False)

            Fp_vox = np.zeros(np.shape(maskvox))
            Fp_vox[maskvox == 1] = np.squeeze(Fp[:])
            Fp_map = ndimage.rotate(np.reshape(Fp_vox, np.shape(mask)), 90, reshape=False)

            mu_cart_vox = np.zeros((np.shape(maskvox)[0], 3))
            mu_cart_vox[maskvox == 1, :] = np.transpose(mu_cart[:])
            mu_cart_map = ndimage.rotate(np.reshape(mu_cart_vox, np.append(np.shape(mask), 3)), 90, reshape=False)

            zslice = pred_slice

            fig = plt.figure()
            plt.imshow(D_par_map[:, :, zslice], vmin=0., vmax=3.)
            D_par_map_nii = nib.Nifti1Image(D_par_map, affine=np.eye(4))
            nib.save(D_par_map_nii, args.save_path + '/volume_'+'_stick_D_' + ' mc sample ' + str(i) + '.nii.gz')
            plt.colorbar()
            fig.suptitle('stick parallel diffusivity ($\mu$m$^2$/ms)' + ' mc sample ' + str(i))
            # plt.xlabel()
            # ax[0, i].xaxis.set_ticklabels([])
            # ax[0, i].set_title('stick parallel diffusivity ($\mu$m$^2$/ms)' + ' mc sample ' + str(i))
            # ax[0, i].axis('off')
            fig_save_name = args.save_path + '/summary_slice_'+str(pred_slice)+'_stick_D_' + ' mc sample ' + str(i) + '.svg'
            plt.savefig(fig_save_name, bbox_inches='tight')

            fig = plt.figure()
            plt.imshow(D_iso_map[:, :, zslice], vmin=0., vmax=3.)
            plt.colorbar()
            D_iso_map_nii = nib.Nifti1Image(D_iso_map, affine=np.eye(4))
            nib.save(D_iso_map_nii, args.save_path + '/volume_'+'_ball_D_' + ' mc sample ' + str(i) + '.nii.gz')
            # ax[1, i].set_title('ball isotropic diffusivity ($\mu$m$^2$/ms)' + ' mc sample ' + str(i))
            # ax[1, i].axis('off')
            fig.suptitle('ball isotropic diffusivity ($\mu$m$^2$/ms)' + ' mc sample ' + str(i))
            fig_save_name = args.save_path + '/summary_slice_'+str(pred_slice)+'_ball_D_' + ' mc sample ' + str(i) + '.svg'
            plt.savefig(fig_save_name, bbox_inches='tight')

            fig = plt.figure()
            plt.imshow(theta_map[:, :, zslice], vmin=0, vmax=3.14)
            plt.colorbar()
            fig.suptitle('theta' + ' mc sample ' + str(i))
            theta_map_nii = nib.Nifti1Image(theta_map, affine=np.eye(4))
            nib.save(theta_map_nii, args.save_path + '/volume_'+'_theta_' + ' mc sample ' + str(i) + '.nii.gz')
            fig_save_name = args.save_path + '/summary_slice_'+str(pred_slice)+'_theta_' + ' mc sample ' + str(i) + '.svg'
            plt.savefig(fig_save_name, bbox_inches='tight')

            fig = plt.figure()
            plt.imshow(phi_map[:, :, zslice], vmin=-3.14, vmax=3.14)
            plt.colorbar()
            fig.suptitle('phi' + ' mc sample ' + str(i))
            phi_map_nii = nib.Nifti1Image(phi_map, affine=np.eye(4))
            nib.save(phi_map_nii, args.save_path + '/volume_'+'_phi_' + ' mc sample ' + str(i) + '.nii.gz')
            fig_save_name = args.save_path + '/summary_slice_'+str(pred_slice)+'_phi_' + ' mc sample ' + str(i) + '.svg'
            plt.savefig(fig_save_name, bbox_inches='tight')

            fig = plt.figure()
            plt.imshow(1 - Fp_map[:, :, zslice], vmin=0., vmax=1.)
            plt.colorbar()
            fp_map_nii = nib.Nifti1Image(Fp_map, affine=np.eye(4))
            nib.save(fp_map_nii, args.save_path + '/volume_'+'_fp_' + ' mc sample ' + str(i) + '.nii.gz')
            fig.suptitle('stick volume fraction' + ' mc sample ' + str(i))
            fig_save_name = args.save_path + '/summary_slice_'+str(pred_slice)+'_stick_volume_fraction_' + ' mc sample ' + str(i) + '.svg'
            plt.savefig(fig_save_name, bbox_inches='tight')

            fig = plt.figure()
            plt.imshow(mu_cart_map[:, :, zslice, :])
            plt.colorbar()
            mu_cart_nii = nib.Nifti1Image(mu_cart_map, affine=np.eye(4))
            nib.save(mu_cart_nii, args.save_path + '/volume_'+'_mu_cart_' + ' mc sample ' + str(i) + '.nii.gz')
            fig.suptitle('Color stick volume fraction' + ' mc sample ' + str(i))
            fig_save_name = args.save_path + '/summary_slice_'+str(pred_slice)+'_color_stick_volume_fraction_' + ' mc sample ' + str(i) + '.svg'
            plt.savefig(fig_save_name, bbox_inches='tight')

    else:
        raise NotImplementedError

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    # fig_save_name = args.save_path + '/summary_slice_'+str(pred_slice)+'.svg'
    # plt.savefig(fig_save_name, bbox_inches='tight')
    # plt.show()

    print('Done')


if __name__ == "__main__":
    # Load the arguments
    args = get_args()
    #pred_slice=30
    #pred_slice=69
    #pred_slice=70
    # pred_slice=90
    predictions = pred(args=args)

