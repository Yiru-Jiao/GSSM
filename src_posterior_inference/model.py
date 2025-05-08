'''
This script defines the GSSM model and loss function for posterior inference of context conditioned proximity distribution.
'''

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils import modules
small_eps = 1e-6


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
    

class GSSM(nn.Module):
    def __init__(self, encoder_selection='all', single_output=None, return_attention=False):
        super(GSSM, self).__init__()
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
        self.combi_encoder = self.define_combi_encoder()
        self.AttentionDecoder = modules.AttentionDecoder(self.final_seq_len, latent_dims=64,
                                                         single_output=single_output,
                                                         return_attention=return_attention)

    def define_combi_encoder(self,):
        if self.encoder_selection==['current'] or self.encoder_selection==['current+acc']:
            def combi_encoder(x):
                x_current = self.CurrentEncoder(x)
                return x_current # (batch_size, 12/13, latent_dims=64)
        elif self.encoder_selection==['current','environment'] or self.encoder_selection==['current+acc','environment']:
            def combi_encoder(x):
                x_current, x_environment = x
                x_current = self.CurrentEncoder(x_current)
                x_environment = self.EnvEncoder(x_environment)
                return torch.cat([x_current, x_environment], dim=1) # (batch_size, 16/17, latent_dims=64)
        elif self.encoder_selection==['current','profiles'] or self.encoder_selection==['current+acc','profiles']:
            def combi_encoder(x):
                x_current, x_ts = x
                x_current = self.CurrentEncoder(x_current)
                x_ts = self.TSEncoder(x_ts)
                return torch.cat([x_current, x_ts], dim=1) # (batch_size, 17/18, latent_dims=64)
        elif self.encoder_selection==['current','environment','profiles'] or self.encoder_selection==['current+acc','environment','profiles']:
            def combi_encoder(x):
                x_current, x_environment, x_ts = x
                x_current = self.CurrentEncoder(x_current)
                x_environment = self.EnvEncoder(x_environment)
                x_ts = self.TSEncoder(x_ts)
                return torch.cat([x_current, x_environment, x_ts], dim=1) # (batch_size, 21/22, latent_dims=64)
        else:
            Warning('Invalid encoder selection.')
        return combi_encoder

    def forward(self, x):
        latent = self.combi_encoder(x)
        out = self.AttentionDecoder(latent)
        return out


class LogNormalNLL(nn.Module):
    '''
    Negative log-likelihood loss function for log-normal distribution.
    '''
    def __init__(self, small_eps=small_eps):
        super(LogNormalNLL, self).__init__()
        self.log2pi = torch.log(torch.tensor(2*3.1415926535897932384626433832795))
        self.small_eps = small_eps

    def forward(self, out, s):
        mu = out[0]
        log_var = out[1]
        log_s = torch.log(torch.clamp(s, min=self.small_eps))
        nll = 0.5 * (self.log2pi + log_var + (log_s-mu)**2 / torch.exp(log_var)) + log_s
        loss = nll.mean()
        return loss


class SmoothLogNormalNLL(nn.Module):
    '''
    Negative log-likelihood loss function for log-normal distribution with smoothness regularization.
    '''
    def __init__(self, beta=5., small_eps=small_eps):
        super(SmoothLogNormalNLL, self).__init__()
        self.beta = beta
        self.log2pi = torch.log(torch.tensor(2*3.1415926535897932384626433832795))
        self.small_eps = small_eps

    def kl_divergence(self, mu1, log_var1, mu2, log_var2):
        return 0.5 * (log_var2 - log_var1 + (torch.exp(log_var1)+(mu1-mu2)**2) / torch.exp(log_var2) - 1)
    
    def js_divergence(self, mu1, log_var1, mu2, log_var2):
        mu = 0.5 * (mu1 + mu2)
        var = 0.5 * (torch.exp(log_var1) + torch.exp(log_var2))
        log_var = torch.log(torch.clamp(var, min=self.small_eps))
        kl1 = self.kl_divergence(mu1, log_var1, mu, log_var)
        kl2 = self.kl_divergence(mu2, log_var2, mu, log_var)
        return 0.5 * (kl1 + kl2)

    def forward(self, out, s, inducing_out):
        mu = out[0]
        log_var = out[1]
        log_s = torch.log(torch.clamp(s, min=self.small_eps))
        nll = 0.5 * (self.log2pi + log_var + (log_s-mu)**2 / torch.exp(log_var)) + log_s

        mu_prime, log_var_prime = inducing_out
        js_divergence = self.js_divergence(mu, log_var, mu_prime, log_var_prime)
        loss = nll.mean() + self.beta*js_divergence.mean()
        return loss
