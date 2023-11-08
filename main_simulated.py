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

from libs.mri_pysical_models import msdki
from libs.mri_pysical_models import ball
from libs.mri_pysical_models import ball_stick
from libs.add_noise_simulation import add_noise

from pathlib import Path


def train(args):
    nvox = 1024
    nclus = 3
    # p = [0.1, 0.1, 0.2, 0.5]
    p = [0.5, 0.4]
    p = np.append(p, 1 - np.sum(p))
    clusters = np.random.choice(range(0, nclus), size=(nvox,), p=p)

    D = [1.0, 1.5, 3.0]
    K = [1.25, 1.0, 0.0]

    mu = np.stack((D, K))
    var = np.diag([args.model.prior_std, args.model.prior_std])
    var = np.stack((var, var, var*0.2))
    # print(var.shape)
    # var = np.diag([0.001, 0.001])

    params = np.zeros((nvox, 2))

    for vox in range(0, nvox):
        params[vox, :] = np.random.multivariate_normal(mu[:, clusters[vox]], var[clusters[vox], :, :])

    # params[params < 0] = 0.01

    directory = '/home/moucheng/projects_data/HCP/103818_1'
    bvals = np.loadtxt(directory + '/T1w/Diffusion/bvals')
    bvals *= 1e-03
    bvecs = np.loadtxt(directory + '/T1w/Diffusion/bvecs')
    bvecs = np.transpose(bvecs)
    grad = np.concatenate((bvecs, bvals[:, None]), axis=1)

    tor_params = torch.from_numpy(params)
    tor_grad = torch.from_numpy(grad)
    tor_grad = tor_grad.to(torch.float32)
    if args.model.mri == 'msdki':
        S = msdki(tor_grad, tor_params)
    # elif args.model.mri == 't2adc':
        # S = t2adc(tor_grad, tor_params)
    else:
        raise NotImplementedError
    S = S.to(torch.float32)

    b_values_no0 = torch.FloatTensor(bvals)
    gradient_directions_no0 = torch.FloatTensor(bvecs)

    if args.model.name == 'mlp':
        net = Net(gradient_directions_no0, b_values_no0, grad, args.model.nparams, args.model.mri)

        model_save_path = args.save_path + '/models_simulated_data/' + args.model.name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = args.model.name + \
                     '_par_' + str(args.model.nparams) + \
                     '_mri_' + str(args.model.mri) + \
                     '_std_' + str(args.model.prior_std) + \
                     '_lr_' + str(args.train.lr) + \
                     '_epoch_' + str(args.train.epochs_no)

        model_save_path = model_save_path + '/' + model_name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

    elif args.model.name == 'gaussian':
        net = Net_VAE(gradient_directions_no0=gradient_directions_no0,
                      b_values_no0=b_values_no0,
                      grad=grad,
                      nparams=args.model.nparams,
                      samples=args.model.samples,
                      mri_model=args.model.mri,
                      prior_std=args.model.prior_std,
                      act=args.model.act)

        model_save_path = args.save_path + '/models_simulated_data/' + args.model.name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = args.model.name + \
                     '_dim_' + str(args.model.samples) + \
                     '_par_' + str(args.model.nparams) + \
                     '_mri_' + str(args.model.mri) + \
                     '_std_' + str(args.model.prior_std) + \
                     '_lr_' + str(args.train.lr) + \
                     '_epoch_' + str(args.train.epochs_no) + \
                     '_alpha_' + str(args.train.alpha) + \
                     '_anneal_' + str(args.train.anneal_rate) + \
                     '_act_' + str(args.model.act) + \
                     '_warm_' + str(args.train.warm_up)

        model_save_path = model_save_path + '/' + model_name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

    else:
        raise NotImplementedError

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=args.train.lr)
    trainloader = utils.DataLoader(S,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=2,
                                   drop_last=True)

    net.train()
    for epoch in range(args.train.epochs_no):
        print("-----------------------------------------------------------------")
        print("Epoch: {}".format(epoch))

        running_loss = 0.
        running_loss_l2 = 0.
        running_loss_kl = 0.
        running_loss_kl_scaled = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):

            optimizer.zero_grad()
            if args.model.name == 'mlp':
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)

            elif args.model.name == 'gaussian':
                # X_pred, D_par_pred, D_iso_pred, mu_pred, Fp_pred, mu, log_var, consistency_loss = net(X_batch)
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)
                running_loss_l2 += loss.item()
                kl_loss = torch.mean(-0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'] ** 2 - outputs['log_var'].exp(), dim=1), dim=0)

                # annealing schedule of kl loss
                if epoch < args.train.warm_up*args.train.epochs_no:
                    alpha = 0.0
                else:
                    alpha = args.train.anneal_rate*(epoch - args.train.warm_up*args.train.epochs_no)
                    alpha = min(alpha, args.train.alpha)

                loss += kl_loss*alpha

                running_loss_kl += kl_loss.item()
                running_loss_kl_scaled += alpha*kl_loss.item()

            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if args.model.name == 'mlp':
            print("Loss: {}".format(running_loss))
        elif args.model.name == 'gaussian':
            print("L2 Loss: {}".format(running_loss_l2))
            print("KL Loss: {}".format(running_loss_kl))
            print("Scaled KL Loss: {}".format(running_loss_kl_scaled))
        else:
            raise NotImplementedError

    print("Training Done")

    save_model_name_full = model_save_path + '/' + model_name + '.pt'
    torch.save(net, save_model_name_full)

    net.eval()
    with torch.no_grad():
        X_pred = net(S)

    # signal_pred = X_pred['signal'].numpy()
    params_pred = X_pred['params'].numpy()

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # for i in range(0, nclus):
    #     ax[0].plot(params[clusters == i, 0], params_pred[clusters == i, 0], 'o', markersize=1)
    #     ax[1].plot(params[clusters == i, 1], params_pred[clusters == i, 1], 'o', markersize=1)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    # plt.show()

    # fig = plt.subplots(1, 2, figsize=(10, 5))
    # plt.plot(grad[:, 3], signal_pred[1, :], 'o')
    # plt.plot(grad[:, 3], S[1, :], 'x')

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    # plt.show()

    plt.figure()
    MSEs = []
    MAEs = []
    MSEs_std = []
    for i in range(0, nclus):
        plt.plot(params[clusters == i, 1], params_pred[clusters == i, 1], 'x', markersize=1)
        # print(params[clusters == i, 1].shape)
        mse = (params[clusters == i, 1] - params_pred[clusters == i, 1])**2
        MSEs.append(mse.mean())
        MSEs_std.append(mse.std())

    print('MSEs of K:')
    print(MSEs)
    print(MSEs_std)
    print('\n')
    print('\n')
    # plt.plot([0.0, max(params[:, 1])], [0, max(params[:, 1])])

    plt.xlabel('GT of Kurtosis')
    plt.ylabel('Pred of Kurtosis')
    plt.legend(['cluster 1', 'cluster 2', 'cluster 3'])
    fig_save_name = model_save_path + '/k.png'
    plt.savefig(fig_save_name, bbox_inches='tight')

    plt.figure()
    MSEs = []
    MSEs_std = []
    for i in range(0, nclus):
        plt.plot(params[clusters == i, 0], params_pred[clusters == i, 0], 'o', markersize=1)
        mse = (params[clusters == i, 0] - params_pred[clusters == i, 0])**2
        MSEs.append(mse.mean())
        MSEs_std.append(mse.std())
    # plt.plot([0.0, max(params[:, 0])], [0, max(params[:, 0])])

    print('MSEs of D:')
    print(MSEs)
    print(MSEs_std)
    print('\n')

    plt.xlabel('GT of Diffusivity')
    plt.ylabel('Pred of Diffusivity')
    plt.legend(['cluster 1', 'cluster 2', 'cluster 3'], fontsize=5)
    fig_save_name = model_save_path + '/d.png'
    plt.savefig(fig_save_name, bbox_inches='tight')


if __name__ == "__main__":
    args = get_args()
    trained_model = train(args=args)

