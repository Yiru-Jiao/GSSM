'''
This script defines the GSSM model and loss function for posterior inference of context conditioned proximity distribution.
'''

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils import modules


def send_x_to_device(x, device):
    if isinstance(x, list):
        return tuple([i.to(device) for i in x])
    else:
        return x.to(device)


class custom_dataset(Dataset): 
    def __init__(self, X):
        self.X = X
        if isinstance(X, tuple):
            def get_length():
                return len(self.X[0])
            def get_item(idx):
                return tuple([torch.from_numpy(x_i[idx]).float() for x_i in self.X])
        else:
            def get_length():
                return len(self.X)
            def get_item(idx):
                return torch.from_numpy(self.X[idx]).float()
        self.get_length = get_length
        self.get_item = get_item

    def __len__(self): 
        return self.get_length()

    def __getitem__(self, idx): 
        return self.get_item(idx)
    

class UnifiedProximity(nn.Module):
    def __init__(self, encoder_selection='all', single_output=None, return_attention=False):
        super(UnifiedProximity, self).__init__()
        if encoder_selection=='all':
            encoder_selection = ['current+acc', 'environment', 'profiles']
        self.encoder_selection = encoder_selection

        # Define encoders and determine the final sequence length
        self.final_seq_len = 0
        if 'current' in encoder_selection:
            self.CurrentEncoder = modules.CurrentEncoder(input_dims=12, output_dims=64)
            self.final_seq_len += 12
        elif 'current+acc' in encoder_selection:
            self.CurrentEncoder = modules.CurrentEncoder(input_dims=13, output_dims=64)
            self.final_seq_len += 13
        else:
            Warning('Current encoder must be selected.')
        if 'environment' in encoder_selection:
            self.EnvEncoder = modules.EnvEncoder(input_dims=27, output_dims=64)
            self.final_seq_len += 4
        if 'profiles' in encoder_selection:
            self.TSEncoder = modules.TSEncoder(input_dims=4, output_dims=64)
            self.final_seq_len += 5
        self.batch_norm = modules.BatchNormModule(self.final_seq_len)
        self.combi_encoder = self.define_combi_encoder()
        self.final_seq_len += self.batch_norm.noise_dim # noise dimensions are added in batch norm for reguralisation
        self.AttentionDecoder = modules.AttentionDecoder(self.final_seq_len, latent_dims=64,
                                                         single_output=single_output,
                                                         return_attention=return_attention)

    def define_combi_encoder(self,):
        if self.encoder_selection==['current'] or self.encoder_selection==['current+acc']:
            def combi_encoder(x):
                x_current = self.CurrentEncoder(x)
                latent = self.batch_norm(x_current)
                return latent # (batch_size, 12/13+noise_dim, latent_dims=64)
        elif self.encoder_selection==['current','environment'] or self.encoder_selection==['current+acc','environment']:
            def combi_encoder(x):
                x_current, x_environment = x
                x_current = self.CurrentEncoder(x_current)
                x_environment = self.EnvEncoder(x_environment)
                latent = self.batch_norm(torch.cat([x_current, x_environment], dim=1))
                return latent # (batch_size, 16/17+noise_dim, latent_dims=64)
        elif self.encoder_selection==['current','profiles'] or self.encoder_selection==['current+acc','profiles']:
            def combi_encoder(x):
                x_current, x_ts = x
                x_current = self.CurrentEncoder(x_current)
                x_ts = self.TSEncoder(x_ts)
                latent = self.batch_norm(torch.cat([x_current, x_ts], dim=1))
                return latent # (batch_size, 17/18+noise_dim, latent_dims=64)
        elif self.encoder_selection==['current','environment','profiles'] or self.encoder_selection==['current+acc','environment','profiles']:
            def combi_encoder(x):
                x_current, x_environment, x_ts = x
                x_current = self.CurrentEncoder(x_current)
                x_environment = self.EnvEncoder(x_environment)
                x_ts = self.TSEncoder(x_ts)
                latent = self.batch_norm(torch.cat([x_current, x_environment, x_ts], dim=1))
                return latent # (batch_size, 21/22+1, latent_dims=64)
        else:
            Warning('Invalid encoder selection.')
        return combi_encoder

    def forward(self, x):
        latent = self.combi_encoder(x)
        out = self.AttentionDecoder(latent)
        return out


class LogNormalNLL(nn.Module):
    def __init__(self, eps=1e-6):
        super(LogNormalNLL, self).__init__()
        self.log2pi = torch.log(torch.tensor(2*3.1415926535897932384626433832795))
        self.eps = eps

    def forward(self, out, s):
        mu = out[0]
        log_var = out[1]
        log_s = torch.log(torch.clamp(s, min=self.eps))
        nll = 0.5 * (self.log2pi + log_var + (log_s-mu)**2 / torch.exp(log_var)) + log_s
        loss = nll.mean()
        return loss


class SmoothLogNormalNLL(nn.Module):
    def __init__(self, beta=5., eps=1e-6):
        super(SmoothLogNormalNLL, self).__init__()
        self.beta = beta
        self.log2pi = torch.log(torch.tensor(2*3.1415926535897932384626433832795))
        self.eps = eps

    def kl_divergence(self, mu1, log_var1, mu2, log_var2):
        return 0.5 * (log_var2 - log_var1 + (torch.exp(log_var1)+(mu1-mu2)**2) / torch.exp(log_var2) - 1)
    
    def js_divergence(self, mu1, log_var1, mu2, log_var2):
        mu = 0.5 * (mu1 + mu2)
        var = 0.5 * (torch.exp(log_var1) + torch.exp(log_var2))
        log_var = torch.log(torch.clamp(var, min=self.eps))
        kl1 = self.kl_divergence(mu1, log_var1, mu, log_var)
        kl2 = self.kl_divergence(mu2, log_var2, mu, log_var)
        return 0.5 * (kl1 + kl2)

    def forward(self, out, s, inducing_out):
        mu = out[0]
        log_var = out[1]
        log_s = torch.log(torch.clamp(s, min=self.eps))
        nll = 0.5 * (self.log2pi + log_var + (log_s-mu)**2 / torch.exp(log_var)) + log_s

        mu_prime, log_var_prime = inducing_out
        js_divergence = self.js_divergence(mu, log_var, mu_prime, log_var_prime)
        loss = nll.mean() + self.beta*js_divergence.mean()
        return loss
