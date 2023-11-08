import numpy as np
import matplotlib.pyplot as plt
# import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import torch.nn.functional as F
from libs.mri_pysical_models import t2_adc, msdki


model='ball_stick'


class Net_VAE(nn.Module):
    def __init__(self,
                 gradient_directions_no0,
                 b_values_no0,
                 act,
                 grad=None,
                 nparams=5,
                 samples=256,
                 prior_std=1.,
                 mri_model='ball_stick'):

        super(Net_VAE, self).__init__()
        # add grad directions, bvals
        self.grad = grad
        self.prior_std = prior_std
        self.sample_no = samples
        self.mri_model = mri_model
        self.gradient_directions_no0 = gradient_directions_no0
        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3):  # 3 fully connected hidden layers
            if act == 'relu':
                self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ReLU()])
            elif act == 'prelu':
                self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.PReLU()])
            elif act == 'softplus':
                self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.Softplus()])
            else:
                raise NotImplementedError

        # self.encoder = nn.Sequential(*self.fc_layers,
        #                              nn.Linear(len(b_values_no0), len(b_values_no0))
        #                              )

        self.encoder = nn.Sequential(*self.fc_layers,
                                     nn.Linear(len(b_values_no0), self.sample_no)
                                     )

        self.mu = nn.Linear(self.sample_no, self.sample_no)
        self.logvar = nn.Linear(self.sample_no, self.sample_no)

        self.gaus2params = nn.Linear(self.sample_no, nparams)

    def forward(self, X):
        '''
        Args:
            X:
        Returns:
            D_par: diffusitivity
            D_iso: another diffusitivity
        '''

        feat = self.encoder(X)
        mu = self.mu(feat)
        logvar = self.logvar(feat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)*self.prior_std
        latent = eps*std + mu

        params = self.gaus2params(latent)
        params = F.softplus(params)

        if self.mri_model == 'ball_stick':

            D_par = torch.clamp(params[:, 0].unsqueeze(1), min=0.001, max=3)
            D_iso = torch.clamp(params[:, 1].unsqueeze(1), min=0.001, max=3)

            Fp = params[:, 4].unsqueeze(1)
            theta = params[:, 2].unsqueeze(1)
            phi = params[:, 3].unsqueeze(1)

            mu_cart = torch.zeros(3, X.size()[0])
            sintheta = torch.sin(theta)
            mu_cart[0, :] = torch.squeeze(sintheta * torch.cos(phi))
            mu_cart[1, :] = torch.squeeze(sintheta * torch.sin(phi))
            mu_cart[2, :] = torch.squeeze(torch.cos(theta))
            X = Fp * torch.exp(-self.b_values_no0 * D_iso) + (1 - Fp) * torch.exp(-self.b_values_no0 * D_par * torch.einsum("ij,jk->ki", self.gradient_directions_no0, mu_cart) ** 2)

            return {'signal': X,
                    'd_par': D_par,
                    'd_iso': D_iso,
                    'mu_cart': mu_cart,
                    'fp': Fp,
                    'mu': mu,
                    'params': params,
                    'latent': latent,
                    'log_var': logvar}

        elif self.mri_model == 't2_adc':
            X = t2_adc(self.grad, params)
            T2 = params[:, 0].unsqueeze(1)
            D = params[:, 1].unsqueeze(1)

            return {'signal': X,
                    't2': T2,
                    'd': D,
                    'params': params,
                    'latent': latent,
                    'mu': mu,
                    'log_var': logvar}

        elif self.mri_model == 'msdki':

            X = msdki(self.grad, params)
            d = params[:, 0].unsqueeze(1)
            k = params[:, 1].unsqueeze(1)

            return {'signal': X,
                    'k': k,
                    'd': d,
                    'latent': latent,
                    'mu': mu,
                    'log_var': logvar,
                    'params': params}

        else:
            raise NotImplementedError
        # (todo): X = model.torch_signal()  # MAKE MODELS A CLASS WITH "TORCH SIGNAL" and "NUMPY SIGNAL"?





