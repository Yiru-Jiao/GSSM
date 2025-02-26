'''
This script defines the encoder for time series
'''

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dims=4, hidden_dims=64, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.LSTMs = nn.ModuleList([nn.LSTM(input_dims, hidden_dims, num_layers=num_layers, batch_first=True)]*5)

    def forward(self, x): # x: (batch_size, 25, feature_dims=4)
        # for per 0.5 sec (5 time steps, in total 5 time blocks)
        for i, lstm in enumerate(self.LSTMs):
            sub_x = x[:, :(i+1)*5, :] # (batch_size, [5, 10, 15, 20, 25], feature_dims)
            h0 = torch.zeros(self.num_layers, sub_x.size(0), self.hidden_dims).to(x.device)
            c0 = torch.zeros(self.num_layers, sub_x.size(0), self.hidden_dims).to(x.device)
            _, (sub_hidden, _) = lstm(sub_x, (h0, c0)) # hidden: (num_layers, batch_size, hidden_dims)
            if i==0:
                hidden = sub_hidden[-1] # (batch_size, hidden_dims)
            else:
                hidden = torch.cat((hidden, sub_hidden[-1]), dim=0) # (5, batch_size, hidden_dims)
        hidden = hidden.permute(1, 0, 2)
        return hidden # (batch_size, 5, hidden_dims)

