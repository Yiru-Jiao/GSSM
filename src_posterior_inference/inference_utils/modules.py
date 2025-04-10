'''
This script defines the encoders and decoder for the GSSM model.
'''

import os
import sys
import glob
import torch
from torch import nn
import pandas as pd
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.clt_model import spclt
from src_encoder_pretraining.ssrl_utils.utils_general import load_tuned_hyperparameters, configure_model


class TSEncoder(nn.Module):
    def __init__(self, device, input_dims=4, output_dims=64):
        super(TSEncoder, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.dist_metric = 'DTW'
        self.spclt_model = spclt(self.input_dims, self.output_dims,
                                 dist_metric=self.dist_metric, device=device)

    # Load a pretrained model
    def load(self, model_selection, device, path_prepared, continue_training=False):
        tuned_params_dir = f'{path_prepared}representation_hyperparameters.csv'
        if os.path.exists(tuned_params_dir):
            tuned_params = pd.read_csv(tuned_params_dir, index_col=0)
        else:
            print(f'****** {tuned_params_dir} not found ******')
        self = load_tuned_hyperparameters(self, tuned_params, model_selection)
        self.repr_dims = self.output_dims
        self.lr = 0.0001 # this learning rate will not be used in posterior inference, but required for configuring spclt
        model_config = configure_model(self, self.input_dims, device)
        self.spclt_model = spclt(**model_config)

        run_dir = f'{path_prepared}trained_models/'
        save_dir = os.path.join(run_dir, f'{model_selection}')
        existing_models = glob.glob(f'{save_dir}/*_net.pth')
        best_model = 'model' + existing_models[0].split('model')[-1].split('_net')[0]
        self.spclt_model.load(f'{save_dir}/{best_model}')
        print(f'Pretrained encoder for profiles loaded: {model_selection}/{best_model}')
        if not continue_training:
            for param in self.spclt_model.net.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        out = self.spclt_model.encode(x)
        return out #(batch_size, 5, repr_dims=64)


class CurrentEncoder(nn.Module):
    '''
    This encoder functions as encoding the representation for each features,
    therefore the features are designed to **not interact** with each other.
    '''
    def __init__(self, input_dims, output_dims=64):
        super(CurrentEncoder, self).__init__()
        self.feature_extractor = self.order_layers(10, output_dims)

    # Define an layer-ordered MLP
    def order_layers(self, num_layers, output_dims):
        ordered_layers = OrderedDict()
        ordered_layers['linear0'] = nn.Linear(2, 8)
        ordered_layers['gelu0'] = nn.GELU()
        ordered_layers['linear1'] = nn.Linear(8, output_dims//2)
        ordered_layers['gelu1'] = nn.GELU()
        ordered_layers['linear2'] = nn.Linear(output_dims//2, output_dims)
        ordered_layers['dropout2'] = nn.Dropout(0.2)
        ordered_layers['gelu2'] = nn.GELU()
        for i in range(3, num_layers-1):
            ordered_layers[f'linear{i}'] = nn.Linear(output_dims, output_dims)
            ordered_layers[f'dropout{i}'] = nn.Dropout(0.2)
            ordered_layers[f'gelu{i}'] = nn.GELU()
        ordered_layers[f'linear{num_layers-1}'] = nn.Linear(output_dims, output_dims)
        return nn.Sequential(ordered_layers)

    # Load a pretrained model
    def load(self, model_selection, device, path_prepared, continue_training=False):
        run_dir = f'{path_prepared}trained_models/'
        save_dir = os.path.join(run_dir, f'{model_selection}')
        best_model = glob.glob(f'{save_dir}/*_encoder.pth')[0]
        state_dict = torch.load(best_model, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        print(f"Pretrained encoder for current features loaded: {best_model.split('trained_models/')[-1]}")
        if not continue_training:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x): # x: (batch_size, 12 or 13)
        features = x.unsqueeze(-1) #(batch_size, 12 or 13, 1)
        noise = torch.zeros_like(features) # add reference to the features
        features = torch.cat([features, noise], dim=-1) # (batch_size, 12 or 13, 2)
        out = self.feature_extractor(features) # (batch_size, 12 or 13, 64)
        return out


class EnvEncoder(nn.Module):
    '''
    This encoder extracts the environment features into 4 representations,
    within each the features are designed to **interact** with each other.
    '''
    def __init__(self, input_dims=27, output_dims=64):
        super(EnvEncoder, self).__init__()
        self.feature_extractor = nn.ModuleList([
            self.order_layers(6, indim, output_dims) for indim in [5, 8, 7, 7]
        ]) # 4 different blocks for the 4 different categorical features

    # Define an layer-ordered MLP
    def order_layers(self, num_layers, input_dims, output_dims):
        ordered_layers = OrderedDict()
        ordered_layers['linear0'] = nn.Linear(input_dims, output_dims//2)
        ordered_layers['gelu0'] = nn.GELU()
        ordered_layers['linear1'] = nn.Linear(output_dims//2, output_dims)
        ordered_layers['dropout1'] = nn.Dropout(0.2)
        ordered_layers['gelu1'] = nn.GELU()
        for i in range(2, num_layers-1):
            ordered_layers[f'linear{i}'] = nn.Linear(output_dims, output_dims)
            ordered_layers[f'dropout{i}'] = nn.Dropout(0.2)
            ordered_layers[f'gelu{i}'] = nn.GELU()
        ordered_layers[f'linear{num_layers-1}'] = nn.Linear(output_dims, output_dims)
        return nn.Sequential(ordered_layers)

    # Load a pretrained model
    def load(self, model_selection, device, path_prepared, continue_training=False):
        run_dir = f'{path_prepared}trained_models/'
        save_dir = os.path.join(run_dir, f'{model_selection}')
        best_model = glob.glob(f'{save_dir}/*_encoder.pth')[0]
        state_dict = torch.load(best_model, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        print(f"Pretrained encoder for environment features loaded: {best_model.split('trained_models/')[-1]}")
        if not continue_training:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x): # x: (batch_size, 27)
        lighting = x[:,0:5] # (batch_size, 5)
        weather = x[:,5:13] # (batch_size, 8)
        surface = x[:,13:20] # (batch_size, 7)
        traffic = x[:,20:27] # (batch_size, 7)
        out = [block(features) for block, features in zip(self.feature_extractor, [lighting, weather, surface, traffic])]
        return torch.stack(out, dim=1) # (batch_size, 4, 64)


class AttentionBlock(nn.Module):
    def __init__(self, input_dims):
        super(AttentionBlock, self).__init__()
        self.linear_q = nn.Sequential(
            nn.Linear(input_dims, input_dims),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(input_dims, input_dims),
        )
        self.linear_v = nn.Sequential(
            nn.Linear(input_dims, input_dims),
        )
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(input_dims)

    def forward(self, x): # x: (batch_size, seq_len, input_dims)
        queries = self.linear_q(x)
        keys = self.linear_k(x)
        values = self.linear_v(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (values.size(-1) ** 0.5)
        attention = torch.nn.functional.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
        attention = self.dropout(attention)
        attended = torch.matmul(attention, values) # (batch_size, seq_len, output_dims)

        attended = self.layer_norm(attended + x)
        return attended, attention
    

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(FeedForwardBlock, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(output_dims, output_dims),
            nn.Dropout(0.2),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x): # x: (batch_size, seq_len, input_dims)
        if self.input_dims == self.output_dims:
            residual = x
        else: # output_dims needs to be larger than and divisible by input_dims
            residual = torch.repeat_interleave(x, self.output_dims//self.input_dims, dim=-1)
        out = self.mlp(x)
        out = self.layer_norm(out + residual)
        return out


class StackedAttention(nn.Module):
    def __init__(self, dims_list, prefix='block'):
        super(StackedAttention, self).__init__()
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(in_dim) for in_dim, _ in dims_list
        ])
        self.feedforward_blocks = nn.ModuleList([
            FeedForwardBlock(in_dim, out_dim) for in_dim, out_dim in dims_list
        ])
        self.prefix = prefix

    def forward(self, x, attention_matrices=None):
        out = x
        for i, (attn_block, ff_block) in enumerate(zip(self.attention_blocks, self.feedforward_blocks)):
            attended, attention = attn_block(out)
            out = ff_block(attended)
            if attention_matrices is not None:
                attention_matrices[f"{self.prefix}_{i}"] = attention
        return out, attention_matrices


class AttentionDecoder(nn.Module):
    '''
    State (Current + Environment) self-attention: (batch_size, 12/13+4, latent_dims=64) -> (batch_size, 12/13+4, hidden_dims=256=64*4)
    TimeSeries self-attention: (batch_size, 5, latent_dims=64) -> (batch_size, 5, hidden_dims=256=64*4)

    Local interaction with CNN: (batch_size, 256, 12~22) -> (batch_size, 16, 12~22)
    Output with MLP: (batch_size, 16*(12~22)) -> (batch_size, 1)
    '''
    def __init__(self, seq_len, latent_dims=64, single_output=None, return_attention=False):
        super(AttentionDecoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dims = latent_dims
        self.single_output = single_output
        self.return_attention = return_attention

        # Define the attention blocks
        self.SelfAttention = StackedAttention([
            (self.latent_dims, self.latent_dims),
            (self.latent_dims, self.latent_dims),
            (self.latent_dims, self.latent_dims*2),
            (self.latent_dims*2, self.latent_dims*2),
            (self.latent_dims*2, self.latent_dims*4),
            (self.latent_dims*4, self.latent_dims*4),
        ])

        # Define the output layers
        self.output_cnn = nn.Sequential( # (batch_size, latent_dims*4=256, seq_len)
            nn.Conv1d(self.latent_dims*4, self.latent_dims, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.latent_dims, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(), # (batch_size, 16*seq_len)
            nn.Linear(16*self.seq_len, 128),
            nn.Dropout(0.2),
            nn.GELU(),
        )
        self.output_mu = nn.Sequential( # (batch_size, 128)
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.output_log_var = nn.Sequential( # (batch_size, 128)
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, state):
        '''
        `state` can include
        current: (batch_size, 12 or 13, latent_dims=64),
        environment: (batch_size, 1, latent_dims=64),
        ts: (batch_size, 5, latent_dims=64),
        which are concatenated as (batch_size, 12~19, latent_dims=64)
        or additionaly include spacing (batch_size, 1) when testing.
        '''
        if self.return_attention:
            attention_matrices = dict()
        else:
            attention_matrices = None
        if self.single_output == 'intensity':
            spacing = state[:,-1,0:1].detach() # (batch_size, 1)
            state = state[:,:-1]
        attended_state, attention_matrices = self.SelfAttention(state, attention_matrices) # (batch_size, seq_len, latent_dims*4=256)
        transposed_state = attended_state.permute(0, 2, 1) # (batch_size, latent_dims*4=256, seq_len)
        out = self.output_cnn(transposed_state) # (batch_size, 16, seq_len) -> (batch_size, 128)
        mu = self.output_mu(out)
        log_var = self.output_log_var(out)
        if self.single_output is None:
            mu = mu.squeeze() # [batch_size]
            log_var = log_var.squeeze()
            if self.return_attention:
                return mu, log_var, (attended_state, attention_matrices) # attended_state: [batch_size, seq_len, latent_dims=64]
                                                           # attention_matrices: {block_i: [batch_size, seq_len, seq_len]}
            else:
                return mu, log_var
        elif self.single_output == 'mu':
            return mu
        elif self.single_output == 'log_var':
            return log_var
        elif self.single_output == 'intensity':
            assert spacing.size() == mu.size(), f'{spacing.size()} != {mu.size()}'
            assert spacing.size() == log_var.size(), f'{spacing.size()} != {log_var.size()}'
            log_p = torch.log(torch.tensor(0.5))
            log_s = torch.log(torch.clamp(spacing, min=1e-6))
            squared2var = torch.sqrt(2*torch.exp(log_var))
            one_minus_cdf = 0.5*(1-torch.erf((log_s-mu)/squared2var))
            max_intensity = log_p / torch.log(torch.clamp(one_minus_cdf, min=1e-6, max=1-1e-6))
            return torch.log10(torch.clamp(max_intensity, min=1.))
