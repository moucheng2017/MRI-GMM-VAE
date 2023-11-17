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
from arguments import get_args
from libs.helpers import get_data, get_grad, img_infor, reshape_img, extract_slice, mask_img, norm_img
from models.base import Net
from models.vae_gau import Net_VAE
import scipy.ndimage as ndimage
from libs.helpers import cart2sphere, sphere2cart
from libs.helpers import direction_average
import yaml
from libs.mri_pysical_models import msdki, t2_adc
from libs.mri_pysical_models import ball
from libs.mri_pysical_models import ball_stick
from libs.add_noise_simulation import add_noise

from pathlib import Path


def main(args):
    
    if args.config_file:
        config_file = args.config_file
    else:
        print("No config file specified")
        
    with open(config_file) as f:
        train_params = yaml.full_load(f)
        
    # read the training parameters from config file:
    prior_std = train_params['model']['prior_std']
    model_ml = train_params['model']['name']
    nparams = train_params['model']['nparams']
    model_mri = train_params['model']['mri']
    activation = train_params['model']['activation']
    mc_samples = train_params['model']['mc_samples']
    num_workers = train_params['train']['num_workers']
    
    lr = train_params['train']['lr']
    epochs = train_params['train']['epochs_no']
    batch = train_params['train']['batch_size']
    alpha = train_params['train']['alpha']
    anneal_rate = train_params['train']['anneal_rate']
    warmup = train_params['train']['warmup']
    
    save_path = train_params['save_path']
    slice_index = train_params['dataset']['slice_index']
    seed = train_params['seed']
    device = train_params['device']
    data_path = train_params['dataset']['data_dir']
    
    nvox = 1024
    nclus = 3
    # p = [0.1, 0.1, 0.2, 0.5]
    p = [0.5, 0.4]
    p = np.append(p, 1 - np.sum(p))
    clusters = np.random.choice(range(0, nclus), size=(nvox,), p=p)

    D = [1.0, 1.5, 3.0]
    K = [1.25, 1.0, 0.0]

    mu = np.stack((D, K))
    var = np.diag([prior_std, 
                   prior_std])
    var = np.stack((var, var, var*0.2))
    # print(var.shape)
    # var = np.diag([0.001, 0.001])

    params = np.zeros((nvox, 2))

    for vox in range(0, nvox):
        params[vox, :] = np.random.multivariate_normal(mu[:, clusters[vox]], var[clusters[vox], :, :])

    # params[params < 0] = 0.01

    # directory = '/home/moucheng/projects_data/HCP/103818_1'
    bvals = np.loadtxt(data_path + '/T1w/Diffusion/bvals')
    bvals *= 1e-03
    bvecs = np.loadtxt(data_path + '/T1w/Diffusion/bvecs')
    bvecs = np.transpose(bvecs)
    grad = np.concatenate((bvecs, bvals[:, None]), axis=1)

    tor_params = torch.from_numpy(params)
    tor_grad = torch.from_numpy(grad)
    tor_grad = tor_grad.to(torch.float32)
    if model_mri == 'msdki':
        S = msdki(tor_grad, tor_params)
    elif args.model.mri == 'ball_stick':
        S = ball_stick(tor_grad, tor_params)
    elif args.model.mri == 'ball':
        S = ball(tor_grad, tor_params)
    elif args.model.mri == 't2_adc':
        S = t2_adc(tor_grad, tor_params)
    else:
        raise NotImplementedError
    S = S.to(torch.float32)

    b_values_no0 = torch.FloatTensor(bvals)
    gradient_directions_no0 = torch.FloatTensor(bvecs)

    if model_ml == 'mlp':
        net = Net(gradient_directions_no0, 
                  b_values_no0, 
                  grad, 
                  nparams, 
                  model_mri).to(device)

        model_save_path = save_path + '/models_simulated_data/' + model_ml
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = model_ml + \
                     '_par_' + str(nparams) + \
                     '_mri_' + str(model_mri) + \
                     '_std_' + str(prior_std) + \
                     '_lr_' + str(lr) + \
                     '_epoch_' + str(epochs)

        model_save_path = model_save_path + '/' + model_name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

    elif model_ml == 'gaussian':
        net = Net_VAE(gradient_directions_no0=gradient_directions_no0,
                      b_values_no0=b_values_no0,
                      grad=grad,
                      nparams=nparams,
                      samples=mc_samples,
                      mri_model=model_mri,
                      prior_std=prior_std,
                      act=activation).to(device)

        model_save_path = save_path + '/models_simulated_data/' + model_ml
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = model_ml + \
                     '_dim_' + str(mc_samples) + \
                     '_par_' + str(nparams) + \
                     '_mri_' + str(model_mri) + \
                     '_std_' + str(prior_std) + \
                     '_lr_' + str(lr) + \
                     '_epoch_' + str(epochs) + \
                     '_alpha_' + str(alpha) + \
                     '_anneal_' + str(anneal_rate) + \
                     '_act_' + str(activation) + \
                     '_warm_' + str(warmup)

        model_save_path = model_save_path + '/' + model_name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

    else:
        raise NotImplementedError

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=lr)
    trainloader = utils.DataLoader(S,
                                   batch_size=batch,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   drop_last=True)

    net.train()
    for epoch in range(epochs):
        print("-----------------------------------------------------------------")
        print("Epoch: {}".format(epoch))

        running_loss = 0.
        running_loss_l2 = 0.
        running_loss_kl = 0.
        running_loss_kl_scaled = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):

            optimizer.zero_grad()
            if model_ml == 'mlp':
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)

            elif model_ml == 'gaussian':
                # X_pred, D_par_pred, D_iso_pred, mu_pred, Fp_pred, mu, log_var, consistency_loss = net(X_batch)
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)
                running_loss_l2 += loss.item()
                kl_loss = torch.mean(-0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'] ** 2 - outputs['log_var'].exp(), dim=1), dim=0)

                # annealing schedule of kl loss
                if epoch < warmup*epochs:
                    alpha_ = 0.0
                else:
                    alpha_ = anneal_rate*(epoch - warmup*epochs)
                    alpha_ = min(alpha_, alpha)

                loss += kl_loss*alpha_

                running_loss_kl += kl_loss.item()
                running_loss_kl_scaled += alpha*kl_loss.item()

            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if model_ml == 'mlp':
            print("Loss: {}".format(running_loss))
        elif model_ml == 'gaussian':
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
    parser = argparse.ArgumentParser(description="Train a simulated MRI model with PyTorch.")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to config file to use for training",
    )
    args = parser.parse_args()
    main(args)

