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
        return [i.to(device) for i in x]
    else:
        return x.to(device)


class custom_dataset(Dataset): 
    def __init__(self, X):
        self.X = X
        if isinstance(X, tuple):
            def get_length():
                return len(self.X[0])
            def get_item(idx):
                return [torch.from_numpy(x_i[idx]).float() for x_i in self.X]
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
    def __init__(self, device, encoder_selection='all', cross_attention=[], return_attention=False, mask_mode=None):
        super(UnifiedProximity, self).__init__()
        self.device = device
        if encoder_selection=='all':
            encoder_selection = ['current', 'environment', 'profiles']
        self.encoder_selection = encoder_selection
        self.cross_attention = cross_attention
        if 'current' in encoder_selection:
            self.current_encoder = modules.current_encoder()
        else:
            Warning('Current encoder must be selected.')
        if 'environment' in encoder_selection:
            self.environment_encoder = modules.environment_encoder()
        if 'profiles' in encoder_selection:
            self.ts_encoder = modules.ts_encoder(device, mask_mode=mask_mode)
        self.attention_decoder = modules.attention_decoder(encoder_selection=self.encoder_selection,
                                                           cross_attention=self.cross_attention,
                                                           return_attention=return_attention)
        self.combi_encoder = self.define_combi_encoder()

    def select_best_model(self, pretraining_evaluation):
        pretraining_evaluation = pretraining_evaluation.copy()
        order_columns = []
        for column in pretraining_evaluation.columns:
            if 'global_' in column:
                if 'dist' in column:
                    pretraining_evaluation[f'order_{column}'] = pretraining_evaluation[column].rank(ascending=True)
                else:
                    pretraining_evaluation[f'order_{column}'] = pretraining_evaluation[column].rank(ascending=False)
                order_columns.append(f'order_{column}')
        pretraining_evaluation['avg_order'] = pretraining_evaluation[order_columns].mean(axis=1)
        best_model = pretraining_evaluation.sort_values(by='avg_order').iloc[0]
        return best_model

    def load_pretrained_encoders(self, path_prepared='../PreparedData/'):
        if 'current' in self.encoder_selection:
            pretraining_evaluation = pd.read_csv(path_prepared + 'EncoderPretraining/current_autoencoder/evaluation.csv')
            best_model = self.select_best_model(pretraining_evaluation)
            self.current_encoder.load(best_model['bslr'], self.device, path_prepared)
        if 'environment' in self.encoder_selection:
            pretraining_evaluation = pd.read_csv(path_prepared + 'EncoderPretraining/environment_autoencoder/evaluation.csv')
            best_model = self.select_best_model(pretraining_evaluation)
            self.environment_encoder.load(best_model['bslr'], self.device, path_prepared)
        if 'profiles' in self.encoder_selection:
            pretraining_evaluation = pd.read_csv(path_prepared + 'EncoderPretraining/spclt/evaluation.csv')
            best_model = self.select_best_model(pretraining_evaluation)
            self.ts_encoder.load(best_model['model'], self.device, path_prepared)

    def define_combi_encoder(self,):
        if self.encoder_selection==['current']:
            def combi_encoder(x):
                x_current = self.current_encoder(x)
                return (x_current,)
        elif self.encoder_selection==['current','environment']:
            def combi_encoder(x):
                x_current, x_environment = x
                x_current = self.current_encoder(x_current)
                x_environment = self.environment_encoder(x_environment)
                return (x_current, x_environment)
        elif self.encoder_selection==['current','profiles']:
            def combi_encoder(x):
                x_current, x_ts = x
                x_current = self.current_encoder(x_current)
                x_ts = self.ts_encoder(x_ts)
                return (x_current, x_ts)
        elif self.encoder_selection==['current','environment','profiles']:
            def combi_encoder(x):
                x_current, x_environment, x_ts = x
                x_current = self.current_encoder(x_current)
                x_environment = self.environment_encoder(x_environment)
                x_ts = self.ts_encoder(x_ts)
                return (x_current, x_environment, x_ts)
        else:
            Warning('Invalid encoder selection.')
        return combi_encoder

    def forward(self, x):
        latent = self.combi_encoder(x)
        out = self.attention_decoder(latent)
        return out # (mu, sigma) if return_attention=False; (mu, sigma, hidden_states) if return_attention=True

    def encode(self, states, batch_size, encoding_window=None):
        contexts, _ = states
        data_loader = DataLoader(custom_dataset(contexts), batch_size=batch_size, shuffle=False)

        hidden_representations = []
        for x in data_loader:
            with torch.no_grad():
                latent = self.combi_encoder(send_x_to_device(x, self.device))
                _, _, hidden_states = self.attention_decoder(latent)
                hidden_representations.append(hidden_states[0])
        hidden_representations = torch.cat(hidden_representations, dim=0) # (n_samples, n_compressed_features)
        return hidden_representations.cpu().numpy()


class LogNormalNLL(nn.Module):
    def __init__(self,):
        super(LogNormalNLL, self).__init__()

    def forward(self, out, y):
        mu, sigma = out
        clipped_y = torch.clamp(y, min=1e-6, max=None) # avoid log(0)
        loss = 0.5*((torch.log(clipped_y)-mu)/sigma)**2 + torch.log(sigma) # log(y) follows a normal distribution
        return (loss * 100).mean() # scale and mean over batch
