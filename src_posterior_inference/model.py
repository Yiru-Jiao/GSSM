'''
'''

import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils import modules


class UnifiedProximity(nn.Module):
    def __init__(self, encoder_selection='all', cross_attention='all', return_attention=False, mask_mode=None):
        super(UnifiedProximity, self).__init__()
        if encoder_selection=='all':
            encoder_selection = ['current', 'environment', 'profiles']
        if cross_attention=='all':
            cross_attention = ['first', 'middle', 'last']
        self.encoder_selection = encoder_selection
        self.cross_attention = cross_attention
        if 'current' in encoder_selection:
            self.current_encoder = modules.current_encoder()
        else:
            Warning('Current encoder must be selected.')
        if 'environment' in encoder_selection:
            self.environment_encoder = modules.environment_encoder()
        if 'profiles' in encoder_selection:
            self.ts_encoder = modules.ts_encoder(mask_mode=mask_mode)
        self.attention_decoder = modules.attention_decoder(encoder_selection=self.encoder_selection,
                                                           cross_attention=self.cross_attention,
                                                           return_attention=return_attention)
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
                return (self.current_encoder(x), None, None)
        elif self.encoder_selection==['current','environment']:
            def combi_encoder(x):
                x_current, x_environment = x
                x_current = self.current_encoder(x_current)
                x_environment = self.environment_encoder(x_environment)
                return (x_current, x_environment, None)
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
        x = self.combi_encoder(x)
        out = self.attention_decoder(x)
        return out # mu, sigma


class LogNormalNLL(nn.Module):
    def __init__(self,):
        super(LogNormalNLL, self).__init__()

    def forward(self, out, y):
        mu, sigma = out
        clipped_y = torch.clamp(y, min=1e-6, max=None) # avoid log(0)
        loss = 0.5*((torch.log(clipped_y)-mu)/sigma)**2 + torch.log(sigma) # log(y) follows a normal distribution
        return loss.mean() # mean over batch
