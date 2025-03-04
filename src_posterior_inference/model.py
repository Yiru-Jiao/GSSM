'''
This script defines the SSSE model and loss function for posterior inference of context conditioned proximity distribution.
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
    def __init__(self, device, encoder_selection='all', return_attention=False):
        super(UnifiedProximity, self).__init__()
        self.device = device
        if encoder_selection=='all':
            encoder_selection = ['current+acc', 'environment', 'profiles']
        self.encoder_selection = encoder_selection
        if 'current' in encoder_selection or 'current+acc' in encoder_selection:
            self.CurrentEncoder = modules.CurrentEncoder(input_dims=1, output_dims=64)
        else:
            Warning('Current encoder must be selected.')
        if 'environment' in encoder_selection:
            self.EnvEncoder = modules.EnvEncoder(input_dims=27, output_dims=64)
        if 'profiles' in encoder_selection:
            self.TSEncoder = modules.TSEncoder(device, input_dims=4, output_dims=64)
        self.AttentionDecoder = modules.AttentionDecoder(encoder_selection=self.encoder_selection,
                                                         return_attention=return_attention)
        self.combi_encoder = self.define_combi_encoder()

    def select_best_model(self, pretraining_evaluation):
        pretraining_evaluation = pretraining_evaluation.copy()
        order_columns = []
        for column in pretraining_evaluation.columns:
            if 'global_' in column or 'local_' in column:
                if 'dist' in column:
                    pretraining_evaluation[f'order_{column}'] = pretraining_evaluation[column].rank(ascending=True)
                else:
                    pretraining_evaluation[f'order_{column}'] = pretraining_evaluation[column].rank(ascending=False)
                order_columns.append(f'order_{column}')
        pretraining_evaluation['avg_order'] = pretraining_evaluation[order_columns].mean(axis=1)
        best_model = pretraining_evaluation.sort_values(by='avg_order').iloc[0]
        return best_model

    def load_pretrained_encoders(self, dataset_name, path_prepared='../PreparedData/', continue_training=False):
        if 'current' in self.encoder_selection:
            path_encoder = path_prepared + f'EncoderPretraining/current_autoencoder/{dataset_name}/'
            pretraining_evaluation = pd.read_csv(path_encoder + 'evaluation.csv')
            best_model = self.select_best_model(pretraining_evaluation)
            self.CurrentEncoder.load(best_model['bslr'], self.device, path_encoder, continue_training)
        if 'current+acc' in self.encoder_selection:
            path_encoder = path_prepared + f'EncoderPretraining/current+acc_autoencoder/{dataset_name}/'
            pretraining_evaluation = pd.read_csv(path_encoder + 'evaluation.csv')
            best_model = self.select_best_model(pretraining_evaluation)
            self.CurrentEncoder.load(best_model['bslr'], self.device, path_encoder, continue_training)
        if 'environment' in self.encoder_selection:
            path_encoder = path_prepared + f'EncoderPretraining/environment_autoencoder/SafeBaseline/'
            pretraining_evaluation = pd.read_csv(path_encoder + 'evaluation.csv')
            best_model = self.select_best_model(pretraining_evaluation)
            self.EnvEncoder.load(best_model['bslr'], self.device, path_encoder, continue_training)
        if 'profiles' in self.encoder_selection:
            path_encoder = path_prepared + f'EncoderPretraining/spclt/{dataset_name}/'
            pretraining_evaluation = pd.read_csv(path_encoder + 'evaluation.csv')
            best_model = self.select_best_model(pretraining_evaluation)
            self.TSEncoder.load(best_model['model'], self.device, path_encoder, continue_training)

    def define_combi_encoder(self,):
        if self.encoder_selection==['current'] or self.encoder_selection==['current+acc']:
            def combi_encoder(x):
                x_current = self.CurrentEncoder(x)
                return (x_current,)
        elif self.encoder_selection==['current','environment'] or self.encoder_selection==['current+acc','environment']:
            def combi_encoder(x):
                x_current, x_environment = x
                x_current = self.CurrentEncoder(x_current)
                x_environment = self.EnvEncoder(x_environment)
                return (x_current, x_environment)
        elif self.encoder_selection==['current','profiles'] or self.encoder_selection==['current+acc','profiles']:
            def combi_encoder(x):
                x_current, x_ts = x
                x_current = self.CurrentEncoder(x_current)
                x_ts = self.TSEncoder(x_ts)
                return (x_current, x_ts)
        elif self.encoder_selection==['current','environment','profiles'] or self.encoder_selection==['current+acc','environment','profiles']:
            def combi_encoder(x):
                x_current, x_environment, x_ts = x
                x_current = self.CurrentEncoder(x_current)
                x_environment = self.EnvEncoder(x_environment)
                x_ts = self.TSEncoder(x_ts)
                return (x_current, x_environment, x_ts)
        else:
            Warning('Invalid encoder selection.')
        return combi_encoder

    def forward(self, x):
        latent = self.combi_encoder(x)
        out = self.AttentionDecoder(latent)
        return out

    def encode(self, states, batch_size):
        contexts, _ = states
        data_loader = DataLoader(custom_dataset(contexts), batch_size=batch_size, shuffle=False)

        hidden_representations = []
        with torch.no_grad():
            for x in data_loader:
                latent = self.combi_encoder(send_x_to_device(x, self.device))
                _, _, hidden_states = self.AttentionDecoder(latent)
                hidden_representations.append(hidden_states[0])
        hidden_representations = torch.cat(hidden_representations, dim=0) # (n_samples, final_seq_len, hidden_dim)
        return hidden_representations.cpu().numpy()


class LogNormalNLL(nn.Module):
    def __init__(self,):
        super(LogNormalNLL, self).__init__()
        '''
        Adding log_2pi is important in this loss as otherwise the loss can go 
        too close to zero where the gradient is too small for effective learning.
        '''
        self.log_2pi = 1.8378770664093453

    def forward(self, out, y):
        mu = out[0]
        log_var = out[1]
        log_y = torch.log(y)
        nll = 0.5 * (self.log_2pi + log_var + (log_y-mu)**2 / torch.exp(log_var))
        loss = nll.mean()
        return loss


class SmoothLogNormalNLL(nn.Module):
    def __init__(self, beta=5.):
        super(SmoothLogNormalNLL, self).__init__()
        self.beta = beta
        self.log_2pi = 1.8378770664093453

    def forward(self, out, y, inducing_out):
        mu = out[0]
        log_var = out[1]
        log_y = torch.log(y)
        nll = 0.5 * (self.log_2pi + log_var + (log_y-mu)**2 / torch.exp(log_var))

        mu_prime, log_var_prime = inducing_out
        kl_divergence = 0.5 * (log_var_prime - log_var + (torch.exp(log_var)+(mu-mu_prime)**2) / torch.exp(log_var_prime) - 1)
        
        loss = nll.mean() + self.beta*kl_divergence.mean()
        return loss
