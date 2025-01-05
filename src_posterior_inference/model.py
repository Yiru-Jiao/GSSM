'''
'''

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils import modules


class UnifiedProximity(nn.Module):
    def __init__(self, encoder_selection='all', mask_mode=None):
        super(UnifiedProximity, self).__init__()
        if encoder_selection=='all':
            encoder_selection = ['current', 'environment', 'profiles']
        self.encoder_selection = encoder_selection
        latent_dims = 0
        if 'current' in encoder_selection:
            self.current_encoder = modules.current_encoder()
            latent_dims += 64
        else:
            Warning('Current encoder must be selected.')
        if 'environment' in encoder_selection:
            self.environment_encoder = modules.environment_encoder()
            latent_dims += 64
        if 'profiles' in encoder_selection:
            self.ts_encoder = modules.ts_encoder(mask_mode=mask_mode)
            latent_dims += 128
        self.cross_attention_decoder = modules.cross_attention_decoder(latent_dims=latent_dims)
        self.combi_encoder = self.define_combi_encoder()

    def load_pretrained_encoders(self, device, path_prepared='../PreparedData/'):
        '''
        current: bs1024_lr0.001
        environment: bs16_lr0.001
        time series: topo_ts2vec
        '''
        if 'current' in self.encoder_selection:
            self.current_encoder.load('bs1024_lr0.001', device, path_prepared)
        if 'environment' in self.encoder_selection:
            self.environment_encoder.load('bs16_lr0.001', device, path_prepared)
        if 'profiles' in self.encoder_selection:
            self.ts_encoder.load('topo_ts2vec', device, path_prepared)

    def define_combi_encoder(self,):
        if self.encoder_selection==['current']:
            def combi_encoder(x):
                return self.current_encoder(x)
        elif self.encoder_selection==['current','environment']:
            def combi_encoder(x):
                x_current, x_environment = x
                x_current = self.current_encoder(x_current)
                x_environment = self.environment_encoder(x_environment)
                return torch.cat([x_current, x_environment], dim=1)
        elif self.encoder_selection==['current','profiles']:
            def combi_encoder(x):
                x_current, x_ts = x
                x_current = self.current_encoder(x_current)
                x_ts = self.ts_encoder(x_ts)
                return torch.cat([x_current, x_ts], dim=1)
        elif self.encoder_selection==['current','environment','profiles']:
            def combi_encoder(x):
                x_current, x_environment, x_ts = x
                x_current = self.current_encoder(x_current)
                x_environment = self.environment_encoder(x_environment)
                x_ts = self.ts_encoder(x_ts)
                return torch.cat([x_current, x_environment, x_ts], dim=1)
        else:
            Warning('Invalid encoder selection.')
        return combi_encoder

    def forward(self, x):
        x = self.combi_encoder(x)
        out = self.cross_attention_decoder(x)  # ([bs, 2], [bs, 1], [bs, 1])
        return out


class TruncatedNLL(nn.Module):
    def __init__(self, k, device):
        super(TruncatedNLL, self).__init__()
        self.k = k
        self.sqrt2 = torch.tensor(np.sqrt(2), dtype=torch.float64).to(device)

    def forward(self, out, y):
        mu, sigma, a = out
        y = torch.log(y) # log(y) follows a normal distribution
        loss = 0.5*((y-mu)/sigma)**2 + torch.log(sigma) + torch.log(1.-torch.erf((a-mu)/sigma/self.sqrt2)) - F.logsigmoid(self.k*(y-a))
        return loss.mean() # mean over batch
