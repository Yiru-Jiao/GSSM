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
        out = self.spclt_model.encode(x, mask='all_true') # out: (batch_size, seq_len, repr_dims)
        # out = self.spclt_model.encode(x, mask='all_true', encoding_window='full_series') # out: (batch_size, repr_dims)
        return out # (batch_size, 20, 64)


class current_encoder(nn.Module):
    def __init__(self, input_dims=9, output_dims=64):
        super(current_encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(1, 32),
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
        x = x.unsqueeze(-1) # (batch_size, 9, 1)
        return self.feature_extractor(x) # (batch_size, 9, 64)


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
        x = x.view(x.size(0), 1, -1) # (batch_size, 1, 27)
        return self.feature_extractor(x) # (batch_size, 1, 64)


class attention_decoder(nn.Module):
    def __init__(self, latent_dims=64, hidden_dims=32, fc_dims=8, encoder_selection=[], cross_attention=[]):
        '''
        State (Current + Environment) self-attention: (batch_size, 10, latent_dims=64) -> (batch_size, 10, hidden_dims=32)
        TimeSeries self-attention: (batch_size, 20, latent_dims=64) -> (batch_size, 20, hidden_dims=32)

        Optional cross-attention, use State to query TimeSeries key-value
        First1sec cross-attention: (batch_size, 10, latent_dims=64) -> (batch_size, 20, hidden_dims=32)
        Middle1sec cross-attention: (batch_size, 10, latent_dims=64) -> (batch_size, 20, hidden_dims=32)
        Last1sec cross-attention: (batch_size, 10, latent_dims=64) -> (batch_size, 20, hidden_dims=32)

        Output self-attention: (batch_size, 30~60, hidden_dims=32) -> (batch_size, 30~60, fc_dims=8)
        
        output: (batch_size, 2)
        '''
        super(attention_decoder, self).__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.fc_dims = fc_dims
        self.encoder_selection = encoder_selection
        self.cross_attention = cross_attention
        self.final_seq_len = 30

        self.Q_state, self.K_state, self.V_state = self.define_head(latent_dims, hidden_dims)
        self.Q_ts, self.K_ts, self.V_ts = self.define_head(latent_dims, hidden_dims)
        if 'profiles' in self.encoder_selection:
            if 'first' in self.cross_attention:
                self.Q_first, self.K_first, self.V_first = self.define_head(latent_dims, hidden_dims)
                self.final_seq_len += 10
            if 'middle' in self.cross_attention:
                self.Q_middle, self.K_middle, self.V_middle = self.define_head(latent_dims, hidden_dims)
                self.final_seq_len += 10
            if 'last' in self.cross_attention:
                self.Q_last, self.K_last, self.V_last = self.define_head(latent_dims, hidden_dims)
                self.final_seq_len += 10
        self.Q_out, self.K_out, self.V_out = self.define_head(hidden_dims, fc_dims)
        self.mlp = nn.Sequential(
            nn.Linear(self.final_seq_len * fc_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def define_head(self, input_dims, output_dims):
        Query = nn.Sequential(
            nn.Linear(input_dims, input_dims//2),
            nn.ReLU(),
            nn.Linear(input_dims//2, output_dims)
        )
        Key = nn.Sequential(
            nn.Linear(input_dims, input_dims//2),
            nn.ReLU(),
            nn.Linear(input_dims//2, output_dims, bias=False)
        )
        Value = nn.Sequential(
            nn.Linear(input_dims, input_dims//2),
            nn.ReLU(),
            nn.Linear(input_dims//2, output_dims, bias=False)
        )
        return Query, Key, Value

    def get_qkv(self, x, Query, Key, Value):
        x = F.layer_norm(x, x.size()[1:])
        queries = Query(x)
        keys = Key(x)
        values = Value(x) # (batch_size, seq_len, output_dims)
        return queries, keys, values

    def head_forward(self, queries, keys, values):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (values.size(-1) ** 0.5)
        attention = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
        out = torch.matmul(attention, values)
        return out # (batch_size, seq_len, output_dims)

    def forward(self, current, environment, ts):
        '''
        current: (batch_size, 9, latent_dims=64)
        environment: (batch_size, 1, latent_dims=64)
        ts: (batch_size, 20, latent_dims=64)
        '''
        out_seq = []
        # State self-attention
        state = torch.cat([current, environment], dim=1) # (batch_size, 10, latent_dims=64)
        q_state, k_state, v_state = self.get_qkv(state, self.Q_state, self.K_state, self.V_state)
        attended_state = self.head_forward(q_state, k_state, v_state) # (batch_size, 10, hidden_dims=32)
        out_seq.append(attended_state)
        if 'profiles' in self.encoder_selection:
            # Cross-attention (optional)
            if 'first' in self.cross_attention:
                q_first, k_first, v_first = self.get_qkv(ts[:, :10], self.Q_first, self.K_first, self.V_first)
                attended_first = self.head_forward(q_state, k_first, v_first) # (batch_size, 10, hidden_dims=32)
                out_seq.append(attended_first)
            if 'middle' in self.cross_attention:
                q_middle, k_middle, v_middle = self.get_qkv(ts[:, 5:15], self.Q_middle, self.K_middle, self.V_middle)
                attended_middle = self.head_forward(q_state, k_middle, v_middle) # (batch_size, 10, hidden_dims=32)
                out_seq.append(attended_middle)
            if 'last' in self.cross_attention:
                q_last, k_last, v_last = self.get_qkv(ts[:, -10:], self.Q_last, self.K_last, self.V_last)
                attended_last = self.head_forward(q_state, k_last, v_last) # (batch_size, 10, hidden_dims=32)
                out_seq.append(attended_last)
            # TimeSeries self-attention
            q_ts, k_ts, v_ts = self.get_qkv(ts, self.Q_ts, self.K_ts, self.V_ts)
            attended_ts = self.head_forward(q_ts, k_ts, v_ts) # (batch_size, 20, hidden_dims=32)
            out_seq.append(attended_ts)
        # Output self-attention
        out_seq = torch.cat(out_seq, dim=1)
        q_out, k_out, v_out = self.get_qkv(out_seq, self.Q_out, self.K_out, self.V_out)
        attended_out = self.head_forward(q_out, k_out, v_out) # (batch_size, final_seq_len, fc_dims=8)
        attended_out = F.layer_norm(attended_out, attended_out.size()[1:])
        out = self.mlp(attended_out.view(attended_out.size(0), -1))

        mu = out[:, 0].unsqueeze(-1)
        sigma = F.softplus(out[:, 1].unsqueeze(-1)) + 1e-6 # avoid zero variance
        return mu, sigma

