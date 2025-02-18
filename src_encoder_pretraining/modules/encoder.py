'''
This script defines the encoders for models.
The TSEncoder is adapted from TS2Vec https://github.com/zhihanyue/ts2vec All adaptations are marked with comments.
Eventually we use LSTM instead of TSEncoder because it's very slow for large-scale data.
'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


#####################
##   LSTM Encoder  ##
#####################

class LSTMEncoder(nn.Module):
    def __init__(self, input_dims=3, hidden_dims=20*64, num_layers=2, single_output=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.single_output = single_output
        self.lstm = nn.LSTM(input_dims, hidden_dims, num_layers, batch_first=True)

    def forward(self, x): # x: (batch_size, seq_length=20, feature_dims=3)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dims).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dims).to(x.device)
        
        # Forward propagate through LSTM
        output, (hidden, cell) = self.lstm(x, (h0, c0)) # hidden: (num_layers, batch_size, hidden_dims)
        if self.single_output:
            # Only return the hidden state of the last LSTM layer
            return hidden[-1].view(x.size(0), x.size(1), -1) # (batch_size, seq_length, hidden_dims//seq_length)
        else:
            # Return all
            return output, (hidden, cell)


#######################
##     TSEncoder     ##
#######################

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
#     for i in range(B):
#         for _ in range(n):
#             t = np.random.randint(T-l+1)
#             res[i, t:t+l] = False
    # in a more efficient way:
    starts = torch.randint(0, T - l + 1, (B, n))
    offsets = torch.arange(l).view(1, 1, l)
    row_idx = torch.arange(B).view(-1, 1, 1).expand(-1, n, l)
    col_idx = starts.unsqueeze(-1) + offsets
    res[row_idx.flatten(), col_idx.flatten()] = False

    return res


def generate_binomial_mask(B, T, p=0.5):
#    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
    # in a more efficient way:
    return torch.rand(B, T) < p


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128, depth=4, mask_mode=None):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        if mask_mode is None:
            self.mask_mode = 'binomial'
        else:
            self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):
        nan_mask = ~torch.isnan(x).any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
#         elif mask == 'all_true':
#             mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
#         elif mask == 'all_false':
#             mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
#         elif mask == 'mask_last':
#             mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
#             mask[:, -1] = False
        # in a more efficient way:
        else:
            B, T = x.shape[:2]
            if mask == 'all_true':
                mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
            elif mask == 'all_false':
                mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            elif mask == 'mask_last':
                mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
                mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
