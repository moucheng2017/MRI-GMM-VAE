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
import scipy.ndimage as ndimage
from libs.helpers import cart2sphere, sphere2cart


def train(args):

    # Load diffusion data:
    directory = args.dataset.data_dir
    bvals = np.loadtxt(directory + '/T1w/Diffusion/bvals')
    bvals *= 1e-03

    bvecs = np.loadtxt(directory + '/T1w/Diffusion/bvecs')
    bvecs = np.transpose(bvecs)

    grad = np.concatenate((bvecs, bvals[:, None]), axis=1)

    # The following will be put into data loader function later ===================================
    img = nib.load(directory + '/T1w/Diffusion/data.nii.gz')
    mask = nib.load(directory + '/T1w/Diffusion/nodif_brain_mask.nii.gz')
    img = img.get_fdata()
    mask = mask.get_fdata()

    nvoxtotal = np.prod(np.shape(img)[0:3])
    nvol = np.shape(img)[3]
    imgvox = np.reshape(img, (nvoxtotal, nvol))

    masktmp = np.zeros(np.shape(mask))
    masktmp[:, :, args.slice_index] = mask[:, :, args.slice_index]
    mask = masktmp
    maskvox = np.reshape(mask, (nvoxtotal))

    imgvoxtofit = imgvox[maskvox == 1]

    normvol = np.where(bvals == min(bvals))

    imgvoxtofitnorm = imgvoxtofit / (np.tile(np.mean(imgvoxtofit[:, normvol], axis=2), (1, nvol)))
    # =============================================================================================

    b_values_no0 = torch.FloatTensor(bvals)
    gradient_directions_no0 = torch.FloatTensor(bvecs)

    if args.model.name == 'mlp':
        net = Net(gradient_directions_no0, b_values_no0, args.model.nparams)
    else:
        raise NotImplementedError

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=args.train.lr)

    if args.debug is True:
        trainloader = utils.DataLoader(torch.from_numpy(imgvoxtofitnorm.astype(np.float32)),
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=2,
                                       drop_last=True)
    else:
        raise NotImplementedError
        # trainloader = utils.DataLoader(dataset,
        #                                batch_size=128,
        #                                shuffle=True,
        #                                num_workers=2,
        #                                drop_last=True)

    num_bad_epochs = 0
    best = 1e-16

    for epoch in range(args.train.epochs_no):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            optimizer.zero_grad()
            X_pred, D_par_pred, D_iso_pred, mu_pred, Fp_pred = net(X_batch)
            loss = criterion(X_pred, X_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Loss: {}".format(running_loss))

        # final_model = net.state_dict()

        # early stopping
        # if running_loss < best:
        #     print("############### Saving good model ###############################")
        #     final_model = net.state_dict()
        #     best = running_loss
        #     num_bad_epochs = 0
        # else:
        #     num_bad_epochs = num_bad_epochs + 1
        #     if num_bad_epochs == args.train.patience:
        #         print("Done, best loss: {}".format(best))
        #         break

    print("Done")

    # Restore best model
    # net.load_state_dict(final_model)
#     return net
#
#
# def eval(net, args):

    net.eval()
    # with torch.no_grad():
    X_real_pred, D_par, D_iso, mu_cart, Fp = net(torch.from_numpy(imgvoxtofitnorm.astype(np.float32)))

    X_real_pred = X_real_pred.numpy()
    D_par = D_par.numpy()
    D_iso = D_iso.numpy()
    mu_cart = mu_cart.numpy()
    Fp = Fp.numpy()

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

    fig, ax = plt.subplots(5, 1, figsize=(5, 20))

    zslice = 70

    plt0 = ax[0].imshow(D_par_map[:, :, zslice])
    plt.colorbar(plt0, ax=ax[0])
    ax[0].xaxis.set_ticklabels([])
    ax[0].set_title('stick parallel diffusivity ($\mu$m$^2$/ms)')
    ax[0].axis('off')

    plt0 = ax[1].imshow(D_iso_map[:, :, zslice])
    plt.colorbar(plt0, ax=ax[1])
    ax[1].set_title('ball isotropic diffusivity ($\mu$m$^2$/ms)')
    ax[1].axis('off')

    plt0 = ax[2].imshow(theta_map[:, :, zslice])
    plt.colorbar(plt0, ax=ax[2])
    ax[2].set_title('theta')
    ax[2].axis('off')

    plt0 = ax[3].imshow(phi_map[:, :, zslice])
    plt.colorbar(plt0, ax=ax[3])
    ax[3].set_title('phi')
    ax[3].axis('off')

    plt0 = ax[4].imshow(1 - Fp_map[:, :, zslice])
    plt.colorbar(plt0, ax=ax[4])
    ax[4].set_title('stick volume fraction')
    ax[4].axis('off')


if __name__ == "__main__":
    args = get_args()
    trained_model = train(args=args)

