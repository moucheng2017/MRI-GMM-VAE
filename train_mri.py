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
from helpers import MRIDataset
from pathlib import Path
import nibabel as nib

def main(args):
    
    if args.config:
        config_file = args.config
    else:
        print("No config file specified. Using default arguments.")

    with open(config_file, "r") as f:
        train_params = yaml.full_load(f)

    # read the training parameters from config file:
    prior_std = train_params['model']['prior_std']
    model_ml = train_params['model']['name']
    nparams = train_params['model']['nparams']
    model_mri = train_params['model']['mri']
    samples = train_params['model']['samples']
    k = train_params['model']['clusters']
    tau = train_params['model']['tau']
    # mc_samples = train_params['model']['mc_samples']

    num_workers = train_params['train']['num_workers']    
    lr = train_params['train']['lr']
    epochs = train_params['train']['epochs_no']
    batch = train_params['train']['batch_size']
    alpha = train_params['train']['alpha']
    anneal_rate = train_params['train']['anneal_rate']
    warmup = train_params['train']['warmup']
    
    save_path = train_params['save_path']
    dataset = train_params['dataset']['name']
    data_path = train_params['dataset']['data_dir']

    # Get the dataset:
    dataset = MRIDataset(data_path)
    trainloader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch, 
                                              shuffle=True, 
                                              num_workers=num_workers)

    b_values_no0 = torch.FloatTensor(trainloader.bvals)
    gradient_directions_no0 = torch.FloatTensor(trainloader.bvecs)
    grad = torch.FloatTensor(trainloader.grad)

    if model_ml == 'mlp':
        if dataset == 'placenta':
            net = Net(gradient_directions_no0, b_values_no0, grad, nparams, model_mri)
        elif dataset == 'hpc':
            net = Net(gradient_directions_no0, b_values_no0, grad, nparams, model_mri)
        else:
            raise NotImplementedError

        model_save_path = args.save_path + '/models_' + dataset + '/' + model_ml + '/' + model_mri
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = model_ml + \
                     '_par_' + str(nparams) + \
                     '_mri_' + str(model_mri) + \
                     '_lr_' + str(lr) + \
                     '_epoch_' + str(epochs)

        model_save_path = model_save_path + '/' + model_name
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

    elif model_ml == 'gaussian':
        if dataset == 'placenta':
            net = Net_VAE(gradient_directions_no0=gradient_directions_no0,
                          b_values_no0=b_values_no0,
                          grad=grad,
                          act=activation,
                          nparams=nparams,
                          samples=samples,
                          mri_model=model_mri,
                          prior_std=prior_std
                          )

        elif dataset == 'hpc':
            net = Net_VAE(gradient_directions_no0=gradient_directions_no0,
                          b_values_no0=b_values_no0,
                          grad=grad,
                          act=activation,
                          nparams=nparams,
                          samples=samples,
                          mri_model=model_mri,
                          prior_std=prior_std)

        else:
            raise NotImplementedError

        model_save_path = save_path + '/models_' + dataset + '/' + model_ml
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = model_ml + \
                     '_dim_' + str(samples) + \
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

    elif model_ml == 'gmm':
        if dataset == 'placenta':
            net = Net_GMM(gradient_directions_no0=gradient_directions_no0,
                          b_values_no0=b_values_no0,
                          grad=grad,
                          act=activation,
                          k=k,
                          tau=tau,
                          nparams=nparams,
                          samples=samples,
                          mri_model=model_mri,
                          prior_std=prior_std
                          )

        elif dataset == 'hpc':
            net = Net_GMM(gradient_directions_no0=gradient_directions_no0,
                          b_values_no0=b_values_no0,
                          grad=grad,
                          k=k,
                          tau=tau,
                          act=activation,
                          nparams=nparams,
                          samples=samples,
                          mri_model=model_mri,
                          prior_std=prior_std)

        else:
            raise NotImplementedError

        model_save_path = save_path + '/models_' + dataset + '/' + model_ml
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        model_name = model_ml + \
                     '_dim_' + str(samples) + \
                     '_par_' + str(nparams) + \
                     '_k_' + str(k) + \
                     '_mri_' + str(model_mri) + \
                     '_std_' + str(prior_std) + \
                     '_lr_' + str(lr) + \
                     '_tau_' + str(tau) + \
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

    # num_bad_epochs = 0
    # best = 1e-16

    best_l2 = 100.

    for epoch in range(epochs):
        print("-----------------------------------------------------------------")
        print("Epoch: {}".format(epoch))
        net.train()
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

            elif model_ml == 'gaussian':
                # X_pred, D_par_pred, D_iso_pred, mu_pred, Fp_pred, mu, log_var, consistency_loss = net(X_batch)
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)
                running_loss_l2 += loss.item()
                kl_loss = torch.mean(-0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'] ** 2 - outputs['log_var'].exp(), dim=1), dim=0)

                # annealing schedule of kl loss
                if epoch < warmup*epochs:
                    alpha = 0.0
                else:
                    alpha = anneal_rate*(epoch - warmup*epochs)
                    alpha = min(alpha, alpha)

                loss += kl_loss*alpha

                running_loss_kl += kl_loss.item()
                running_loss_kl_scaled += alpha*kl_loss.item()

            elif model_ml == 'gmm':
                # X_pred, D_par_pred, D_iso_pred, mu_pred, Fp_pred, mu, log_var, consistency_loss = net(X_batch)
                outputs = net(X_batch)
                loss = criterion(outputs['signal'], X_batch)
                running_loss_l2 += loss.item()
                kl_loss = torch.mean(-0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'] ** 2 - outputs['log_var'].exp(), dim=1), dim=0)

                qy = F.softmax(outputs['y_logits'], dim=-1)
                log_q = F.log_softmax(outputs['y_logits'], dim=-1)
                target = torch.ones(1)*(1 / args.model.k)
                kl_loss2 = -torch.mean(torch.sum(qy * (log_q - torch.log(target)), dim=-1))

                # annealing schedule of kl loss
                if epoch < warmup*epochs:
                    alpha_ = 0.0
                else:
                    alpha_ = anneal_rate*(epoch - warmup*epochs)
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

            if running_loss_l2 < best_l2:
                save_model_name_full = model_save_path + '/' + model_name + '_best.pt'
                torch.save(net, save_model_name_full)
                best_l2 = running_loss_l2

        if model_ml == 'mlp':
            print("Loss: {}".format(running_loss))
        elif model_ml == 'gaussian':
            print("L2 Loss: {}".format(running_loss_l2))
            print("KL Loss: {}".format(running_loss_kl))
            print("Scaled KL Loss: {}".format(running_loss_kl_scaled))
        elif model_ml == 'gmm':
            print("L2 Loss: {}".format(running_loss_l2))
            print("KL Loss: {}".format(running_loss_kl))
            print("Scaled KL Loss: {}".format(running_loss_kl_scaled))
            print("KL Loss 2: {}".format(running_loss_kl2))
            print("Scaled KL Loss 2: {}".format(running_loss_kl_scaled2))
        else:
            raise NotImplementedError

    print("Done")

    save_model_name_full = model_save_path + '/' + model_name + '.pt'
    torch.save(net, save_model_name_full)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep variational MRI model on real-data with PyTorch.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file to use for training",
    )
    args = parser.parse_args()
    main(args)

