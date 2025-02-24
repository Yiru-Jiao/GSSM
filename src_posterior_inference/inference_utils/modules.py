'''
This script defines the encoders and decoder for the SSSE model.
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


class TSEncoder(nn.Module):
    def __init__(self, device, input_dims=4, output_dims=128):
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
        model_config = configure_model(self, self.input_dims, device)
        self.spclt_model = spclt(**model_config)

        run_dir = f'{path_prepared}trained_models/'
        save_dir = os.path.join(run_dir, f'{model_selection}')
        existing_models = glob.glob(f'{save_dir}/*_net.pth')
        best_model = 'model' + existing_models[0].split('model')[-1].split('_net')[0]
        self.spclt_model.load(f'{save_dir}/{best_model}')
        print(f'Pretrained encoder for profiles loaded: {best_model}')
        if not continue_training:
            for param in self.net.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        out = self.spclt_model.encode(x) # out: (batch_size, seq_len=20, repr_dims=128)
        return out # (batch_size, 20, 128)


class CurrentEncoder(nn.Module):
    '''
    This encoder functions as encoding the token for each features,
    therefore the features are designed to **not interact** with each other.
    '''
    def __init__(self, input_dims=1, output_dims=128):
        super(CurrentEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, output_dims//2),
            nn.GELU(),
            nn.Linear(output_dims//2, output_dims),
            nn.GELU(),
            nn.Linear(output_dims, output_dims),
            nn.GELU(),
            nn.Dropout(0.1),
        )

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

    def forward(self, x): # x: (batch_size, 15 or 16)
        features = x.unsqueeze(-1) #(batch_size, 15 or 16, 1)
        out = self.feature_extractor(features) # (batch_size, 15 or 16, 128)
        return out


class EnvEncoder(nn.Module):
    '''
    This encoder needs to extract the environment features as one single token,
    therefore the features are designed to **interact** with each other.
    '''
    def __init__(self, input_dims=27, output_dims=128):
        super(EnvEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, output_dims//2),
            nn.GELU(),
            nn.Linear(output_dims//2, output_dims),
            nn.GELU(),
            nn.Linear(output_dims, output_dims),
            nn.GELU(),
            nn.Dropout(0.1),
        )

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
        out = self.feature_extractor(x) # (batch_size, 128)
        return  out.unsqueeze(1) # (batch_size, 1, 128)


class AttentionBlock(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(AttentionBlock, self).__init__()
        self.linear_q = nn.Linear(input_dims, output_dims)
        self.linear_k = nn.Linear(input_dims, output_dims)
        self.linear_v = nn.Linear(input_dims, output_dims)

        # Dropout layers
        self.attn_dropout = nn.Dropout(0.1)
        self.out_dropout = nn.Dropout(0.1)
        
        # LayerNorm applied after adding the residual connection
        self.layernorm = nn.LayerNorm(output_dims)

        # If input and output dimensions differ, project residual to match output_dims
        if input_dims != output_dims:
            self.residual_projection = nn.Linear(input_dims, output_dims)
        else:
            self.residual_projection = None
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dims)
        residual = x
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        # Compute projections for queries, keys, and values
        queries = self.linear_q(x)
        keys = self.linear_k(x)
        values = self.linear_v(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (values.size(-1) ** 0.5)
        attention = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
        attention = self.attn_dropout(attention)  # dropout on attention weights
        attended = torch.matmul(attention, values) # (batch_size, seq_len, output_dims)
        attended = self.out_dropout(attended)  # dropout on output
        
        # Add residual connection and apply layer normalization
        out = self.layernorm(attended + residual)
        return out, attention


class StackedAttention(nn.Module):
    def __init__(self, dims_list, prefix='block'):
        super(StackedAttention, self).__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(in_dim, out_dim) for in_dim, out_dim in dims_list
        ])
        self.prefix = prefix

    def forward(self, x, attention_matrices=None):
        for i, block in enumerate(self.blocks):
            out, attention = block(x)
            if attention_matrices is not None:
                attention_matrices[f"{self.prefix}_{i+1}"] = attention
        return out, attention_matrices


class AttentionDecoder(nn.Module):
    '''
    State (Current + Environment) self-attention: (batch_size, 15/16+1, latent_dims=128) -> (batch_size, 15/16+1, hidden_dims=64)
    TimeSeries self-attention: (batch_size, 20, latent_dims=128) -> (batch_size, 20, hidden_dims=64)

    Output self-attention: (batch_size, 15~54, hidden_dims=64) -> (batch_size, 15~54, 1)    
    Output with linear: (batch_size, 15~54, 1) -> (batch_size, 2)
    '''
    def __init__(self, latent_dims=128, hidden_dims=64, encoder_selection=[], return_attention=False):
        super(AttentionDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.encoder_selection = encoder_selection
        self.return_attention = return_attention
        
        # Determine the final sequence length
        self.final_seq_len = 0
        if 'current' in self.encoder_selection:
            self.final_seq_len += 15
        if 'current+acc' in self.encoder_selection:
            self.final_seq_len += 16
        if 'environment' in self.encoder_selection:
            self.final_seq_len += 1
        if 'profiles' in self.encoder_selection:
            self.final_seq_len += 20

        # Define the attention blocks
        self.StateAttention = StackedAttention([
            # (self.latent_dims, self.latent_dims),
            # (self.latent_dims, self.latent_dims),
            (self.latent_dims, self.hidden_dims),
            # (self.hidden_dims, self.hidden_dims),
        ], prefix='state')
        self.TSAttention = StackedAttention([
            # (self.latent_dims, self.latent_dims),
            # (self.latent_dims, self.latent_dims),
            (self.latent_dims, self.hidden_dims),
            # (self.hidden_dims, self.hidden_dims),
        ], prefix='ts')
        self.OutAttention = StackedAttention([
            # (self.hidden_dims, self.hidden_dims),
            (self.hidden_dims, self.hidden_dims//4),
            (self.hidden_dims//4, 1),
        ], prefix='out')

        # Define the combi_decoder and final linear layer
        self.combi_decoder = self.define_combi_decoder()
        self.linear = nn.Sequential( # (batch_size, final_seq_len, 1)
            nn.Flatten(1), # (batch_size, final_seq_len)
            nn.Linear(self.final_seq_len, 2),
        ) # (batch_size, 2)

    def define_combi_decoder(self,):
        if self.encoder_selection==['current'] or self.encoder_selection==['current+acc']:
            def combi_decoder(x_tuple):
                state = x_tuple[0] # (batch_size, 15 or 16, latent_dims=128)
                attention_matrices = dict()
                attended_state, attention_matrices = self.StateAttention(state, attention_matrices)
                attended_out, attention_matrices = self.OutAttention(attended_state, attention_matrices)
                out = self.linear(attended_out)
                if self.return_attention:
                    attention_matrices['linear_w'] = self.linear[1].weight
                    attention_matrices['linear_b'] = self.linear[1].bias
                return out, (attended_out, attention_matrices)
        elif self.encoder_selection==['current','environment'] or self.encoder_selection==['current+acc','environment']:
            def combi_decoder(x_tuple):
                current, environment = x_tuple
                attention_matrices = dict()
                state = torch.cat([current, environment], dim=1) # (batch_size, 16 or 17, latent_dims=128)
                attended_state, attention_matrices = self.StateAttention(state, attention_matrices)
                attended_out, attention_matrices = self.OutAttention(attended_state, attention_matrices)
                out = self.linear(attended_out)
                if self.return_attention:
                    attention_matrices['linear_w'] = self.linear[1].weight
                    attention_matrices['linear_b'] = self.linear[1].bias
                return out, (attended_out, attention_matrices)
        elif self.encoder_selection==['current','profiles'] or self.encoder_selection==['current+acc','profiles']:
            def combi_decoder(x_tuple):
                state, ts = x_tuple
                attention_matrices = dict()
                attended_state, attention_matrices = self.StateAttention(state, attention_matrices)
                attended_ts, attention_matrices = self.TSAttention(ts, attention_matrices)
                out_seq = torch.cat([attended_state, attended_ts], dim=1) # (batch_size, 35 or 36, hidden_dims=64)
                attended_out, attention_matrices = self.OutAttention(out_seq, attention_matrices)
                out = self.linear(attended_out)
                if self.return_attention:
                    attention_matrices['linear_w'] = self.linear[1].weight
                    attention_matrices['linear_b'] = self.linear[1].bias
                return out, (attended_out, attention_matrices)
        elif self.encoder_selection==['current','environment','profiles'] or self.encoder_selection==['current+acc','environment','profiles']:
            def combi_decoder(x_tuple):
                current, environment, ts = x_tuple
                attention_matrices = dict()
                state = torch.cat([current, environment], dim=1)
                attended_state, attention_matrices = self.StateAttention(state, attention_matrices)
                attended_ts, attention_matrices = self.TSAttention(ts, attention_matrices)
                out_seq = torch.cat([attended_state, attended_ts], dim=1) # (batch_size, 36 or 37, hidden_dims=64)
                attended_out, attention_matrices = self.OutAttention(out_seq, attention_matrices)
                out = self.linear(attended_out)
                if self.return_attention:
                    attention_matrices['linear_w'] = self.linear[1].weight
                    attention_matrices['linear_b'] = self.linear[1].bias
                return out, (attended_out, attention_matrices)
        return combi_decoder

    def forward(self, x_tuple):
        '''
        current: (batch_size, 15 or 16, latent_dims=128)
        environment: (batch_size, 1, latent_dims=128)
        ts: (batch_size, 20, latent_dims=128)
        '''
        out, hidden_states = self.combi_decoder(x_tuple)
        mu = out[:, 0].unsqueeze(-1)
        sigma = F.softplus(out[:, 1].unsqueeze(-1)) + 1e-6 # add small value to avoid near-zero variance
        if self.return_attention:
            return mu, sigma, hidden_states
        else:
            return mu, sigma

