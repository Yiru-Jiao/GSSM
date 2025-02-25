'''
This script defines the encoder for time series
'''

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dims=4, hidden_dims=3*128, num_layers=2, single_output=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.single_output = single_output
        self.lstm = nn.LSTM(input_dims, hidden_dims, num_layers, batch_first=True)

    def forward(self, x): # x: (batch_size, seq_length, feature_dims=4)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dims).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dims).to(x.device)
        
        # Forward propagate through LSTM
        output, (hidden, cell) = self.lstm(x, (h0, c0)) # hidden: (num_layers, batch_size, hidden_dims)
        if self.single_output:
            # Only return the hidden state of the last LSTM layer
            return hidden[-1].view(x.size(0), 3, -1) # (batch_size, 3, hidden_dims//3)
        else:
            # Return all
            return output, (hidden, cell)

