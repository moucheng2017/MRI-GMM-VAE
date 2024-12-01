import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
from models.baseline import Net
from models.vae import Net_VAE
from models.vae_gmm import Net_GMM
import yaml
import torch.nn.functional as F
from models.mri_pysical_models import msdki, t2_adc
from models.mri_pysical_models import ball
from models.mri_pysical_models import ball_stick
from pathlib import Path


def main(args):
    
    if args.config:
        config_file = args.config
    else:
        print("No config file specified")
        
    with open(config_file) as f:
        train_params = yaml.full_load(f)
        
    # read the training parameters from config file:
    prior_std = train_params['model']['prior_std']
    model_ml = train_params['model']['name']
    nparams = train_params['model']['nparams']
    model_mri = train_params['model']['mri']
    num_workers = train_params['train']['num_workers']
    latent_dim = train_params['model']['latent_dim']
    
    lr = train_params['train']['lr']
    epochs = train_params['train']['epochs_no']
    batch = train_params['train']['batch_size']
    alpha = train_params['train']['alpha']
    anneal_rate = train_params['train']['anneal_rate']
    warmup = train_params['train']['warmup']
    
    save_path = train_params['save_path']
    # seed = train_params['seed']
    data_path = train_params['dataset']['data_dir']
    k = train_params['model']['k']
    
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
                  model_mri)

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

    elif model_ml == 'vae':
        net = Net_VAE(gradient_directions_no0=gradient_directions_no0,
                      b_values_no0=b_values_no0,
                      grad=grad,
                      nparams=nparams,
                      latent_dim=latent_dim,
                      mri_model=model_mri,
                      prior_std=prior_std)

        model_save_path = save_path + '/models_simulated_data/' + model_ml
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = model_ml + \
                     '_dim_' + str(latent_dim) + \
                     '_par_' + str(nparams) + \
                     '_mri_' + str(model_mri) + \
                     '_std_' + str(prior_std) + \
                     '_lr_' + str(lr) + \
                     '_epoch_' + str(epochs) + \
                     '_alpha_' + str(alpha) + \
                     '_anneal_' + str(anneal_rate) + \
                     '_warm_' + str(warmup)

        model_save_path = model_save_path + '/' + model_name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    elif model_ml == 'gm_vae':
        net = Net_GMM(gradient_directions_no0=gradient_directions_no0,
                      b_values_no0=b_values_no0,
                      grad=grad,
                      nparams=nparams,
                      latent_dim=latent_dim,
                      mri_model=model_mri,
                      k=k,
                      prior_std=prior_std)

        model_save_path = save_path + '/models_simulated_data/' + model_ml
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = model_ml + \
                     '_dim_' + str(latent_dim) + \
                     '_par_' + str(nparams) + \
                     '_mri_' + str(model_mri) + \
                     '_k_' + str(k) + \
                     '_std_' + str(prior_std) + \
                     '_lr_' + str(lr) + \
                     '_epoch_' + str(epochs) + \
                     '_alpha_' + str(alpha) + \
                     '_anneal_' + str(anneal_rate) + \
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
        running_loss_kl2 = 0.
        running_loss_kl_scaled2 = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):

            optimizer.zero_grad()
            if model_ml == 'mlp':
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)

            elif model_ml == 'vae':
                # X_pred, D_par_pred, D_iso_pred, mu_pred, Fp_pred, mu, log_var, consistency_loss = net(X_batch)
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)
                running_loss_l2 += loss.item()
                kl_loss = torch.mean(-0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'] ** 2 - outputs['log_var'].exp(), dim=1), dim=0)

                # annealing schedule of kl loss
                if epoch < 0.1*epochs:
                    alpha_ = 0.0
                else:
                    alpha_ = anneal_rate*(0.99*epochs)
                    alpha_ = min(alpha_, alpha)

                loss += kl_loss*alpha_

                running_loss_kl += kl_loss.item()
                running_loss_kl_scaled += alpha*kl_loss.item()

            elif model_ml == 'gm_vae':
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)
                running_loss_l2 += loss.item()
                kl_loss = torch.mean(-0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'] ** 2 - outputs['log_var'].exp(), dim=1), dim=0)

                qy = F.softmax(outputs['y_logits'], dim=-1)
                log_q = F.log_softmax(outputs['y_logits'], dim=-1)

                k = outputs['y_logits'].size(-1)  # Number of categories
                target = torch.full_like(qy, 1 / k)  # Shape-matched uniform distribution

                # Compute KL divergence
                kl_loss2 = torch.mean(torch.sum(qy * (log_q - torch.log(target)), dim=-1))

                # annealing schedule of kl loss
                if epoch < 0.1*epochs:
                    alpha_ = 0.0
                else:
                    alpha_ = anneal_rate*(0.99*epochs)
                    alpha_ = min(alpha, alpha_)

                loss += kl_loss*alpha_ + kl_loss2*alpha_

                running_loss_kl += kl_loss.item()
                running_loss_kl_scaled += alpha_*kl_loss.item()

                running_loss_kl2 += kl_loss2.item()
                running_loss_kl_scaled2 += alpha_*kl_loss2.item()

            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if model_ml == 'mlp':
            print("Loss: {}".format(running_loss))
        elif model_ml == 'vae':
            print("L2 Loss: {}".format(running_loss_l2))
            print("KL Loss: {}".format(running_loss_kl))
            print("Scaled KL Loss: {}".format(running_loss_kl_scaled))
        elif model_ml == 'gm_vae':
            print("L2 Loss: {}".format(running_loss_l2))
            print("KL Loss: {}".format(running_loss_kl))
            print("Scaled KL Loss: {}".format(running_loss_kl_scaled))
            print("KL Loss2: {}".format(running_loss_kl2))
            print("Scaled KL Loss2: {}".format(running_loss_kl_scaled2))    
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

    plt.figure()
    MSEs = []
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
    plt.legend(['cluster 1', 'cluster 2', 'cluster 3'], fontsize=10)
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
    plt.legend(['cluster 1', 'cluster 2', 'cluster 3'], fontsize=10)
    fig_save_name = model_save_path + '/d.png'
    plt.savefig(fig_save_name, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simulated MRI model with PyTorch.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file to use for training",
    )
    args = parser.parse_args()
    main(args)

