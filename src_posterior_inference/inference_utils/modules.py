'''
'''

import os
import sys
import glob
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.clt_model import spclt
from src_encoder_pretraining.ssrl_utils.utils_general import load_tuned_hyperparameters, configure_model


class ts_encoder(nn.Module):
    def __init__(self, input_dims=3, output_dims=64, mask_mode=None):
        super(ts_encoder, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = 24
        self.dist_metric = 'DTW'
        self.mask_mode = mask_mode
        self.spclt_model = spclt(input_dims, output_dims, hidden_dims=64, dist_metric='DTW', mask_mode=mask_mode)
        self.feature_extractor = self.spclt_model.net

    # Load a pretrained model
    def load(self, model_selection, device, path_prepared='../PreparedData/'):
        tuned_params_dir = f'{path_prepared}EncoderPretraining/spclt/representation_hyperparameters.csv'
        if os.path.exists(tuned_params_dir):
            tuned_params = pd.read_csv(tuned_params_dir, index_col=0)
        else:
            print(f'****** {tuned_params_dir} not found ******')
        self = load_tuned_hyperparameters(self, tuned_params, model_selection)
        model_config = configure_model(self, self.input_dims, device)
        self.spclt_model = spclt(**model_config)

        run_dir = f'{path_prepared}EncoderPretraining/spclt/trained_models/'
        save_dir = os.path.join(run_dir, f'{model_selection}')
        existing_models = glob.glob(f'{save_dir}/*_net.pth')
        best_model = 'model' + existing_models[0].split('model')[-1].split('_net')[0]
        self.spclt_model.load(f'{save_dir}/{best_model}')
        print(f'Loaded the best model: {best_model}')
        self.feature_extractor = self.spclt_model.net
        
    def forward(self, x):
        out = self.feature_extractor(x, mask='all_true') # out: (batch_size, seq_len, repr_dims)
        out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2).squeeze(1) # out: (batch_size, repr_dims)
        return out


class current_encoder(nn.Module):
    def __init__(self, input_dims=9, output_dims=64):
        super(current_encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, output_dims),
            nn.ReLU(),
            nn.Linear(output_dims, output_dims),
            nn.ReLU(),
        )

    # Load a pretrained model
    def load(self, model_selection, device, path_prepared='../PreparedData/'):
        run_dir = f'{path_prepared}EncoderPretraining/current_autoencoder/trained_models/'
        save_dir = os.path.join(run_dir, f'{model_selection}')
        best_model = glob.glob(f'{save_dir}/*_encoder.pth')[0]
        state_dict = torch.load(best_model, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        print(f'Loaded the best model: {best_model.split(model_selection)[-1]}')

    def forward(self, x):
        return self.feature_extractor(x)


class environment_encoder(nn.Module):
    def __init__(self, input_dims=27, output_dims=64):
        super(environment_encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, output_dims),
            nn.ReLU(),
            nn.Linear(output_dims, output_dims),
            nn.ReLU(),
        )

    # Load a pretrained model
    def load(self, model_selection, device, path_prepared='../PreparedData/'):
        run_dir = f'{path_prepared}EncoderPretraining/environment_autoencoder/trained_models/'
        save_dir = os.path.join(run_dir, f'{model_selection}')
        best_model = glob.glob(f'{save_dir}/*_encoder.pth')[0]
        state_dict = torch.load(best_model, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        print(f'Loaded the best model: {best_model.split(model_selection)[-1]}')

    def forward(self, x):
        return self.feature_extractor(x)


class cross_attention_decoder(nn.Module):
    def __init__(self, latent_dims=256, attention_dims=32, fc_dims=8):
        '''
        2 heads, 1 outputs (mu, variance), 1 outputs (a)
        '''
        super(cross_attention_decoder, self).__init__()
        self.attention_dims = attention_dims
        self.fc_dims = fc_dims
        # Shared head for (mu, variance)
        weights_mu_var = self.define_head(latent_dims, attention_dims, fc_dims, 2)
        self.Query_mu_var, self.Key_mu_var, self.Value_mu_var, self.Output_mu_var = weights_mu_var
        # Head for lower bound
        weights_low = self.define_head(latent_dims, attention_dims, fc_dims, 1)
        self.Query_low, self.Key_low, self.Value_low, self.Output_low = weights_low

    def define_head(self, latent_dims, attention_dims, fc_dims, output_dims):
        Query = nn.Sequential(
            nn.Linear(fc_dims, latent_dims),
            nn.ReLU(),
            nn.Linear(latent_dims, attention_dims),
            nn.ReLU(),
        )
        Key = nn.Sequential(
            nn.Linear(latent_dims, latent_dims//2),
            nn.ReLU(),
            nn.Linear(latent_dims//2, attention_dims),
            nn.ReLU(),
        )
        Value = nn.Sequential(
            nn.Linear(latent_dims, latent_dims//2),
            nn.ReLU(),
            nn.Linear(latent_dims//2, attention_dims),
            nn.ReLU(),
        )
        Output = nn.Sequential(
            nn.Linear(attention_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, output_dims),
        )
        return Query, Key, Value, Output

    def head_forward(self, x, Query, Key, Value, Output):
        '''
        x: (batch_size, latent_dims)
        '''
        q0 = torch.zeros(x.size(0), self.fc_dims).to(x.device) # (batch_size, fc_dims)
        queries = Query(q0).unsqueeze(-1)                      # (batch_size, attention_dims, 1)
        keys = Key(x).unsqueeze(1)                             # (batch_size, 1, attention_dims)
        values = Value(x).unsqueeze(-1)                        # (batch_size, attention_dims, 1)
        scores = torch.matmul(queries, keys) / (self.attention_dims ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)          # (batch_size, attention_dims, attention_dims)
        attended = torch.matmul(attention_weights, values)     # (batch_size, attention_dims, 1)
        out = Output(attended.squeeze(-1))                     # (batch_size, output_dims)
        return out

    def forward(self, x):
        '''
        x: (batch_size, latent_dims)
        '''
        # Shared head for (mu, variance)
        mu_var = self.head_forward(x,
            self.Query_mu_var,
            self.Key_mu_var,
            self.Value_mu_var,
            self.Output_mu_var,
        )                                          # (batch_size, 2)
        # Head for lower bound
        low = self.head_forward(x,
            self.Query_low,
            self.Key_low,
            self.Value_low,
            self.Output_low,
        )                                          # (batch_size, 1)
        mu = mu_var[:, 0].unsqueeze(-1)
        sigma = F.softplus(mu_var[:, 1].unsqueeze(-1)) + 1e-6 # avoid zero variance
        a = mu - F.softplus(low)
        return mu, sigma, a

