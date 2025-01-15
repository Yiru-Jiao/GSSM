'''
This script defines the autoencoder model for encoders for current and environment features.
'''

import os
import sys
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import ssrl_utils.utils_data as datautils
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils.modules import current_encoder, environment_encoder


class shared_decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(shared_decoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, output_dims),
        )

    def forward(self, x):
        return self.feature_extractor(x)


class model(nn.Module):
    def __init__(self, encoder_name):
        super(model, self).__init__()
        if encoder_name == 'current':
            self.encoder = current_encoder(input_dims=9, output_dims=64)
            self.decoder = shared_decoder(input_dims=9*64, output_dims=9)
        elif encoder_name == 'environment':
            self.encoder = environment_encoder(input_dims=27, output_dims=64)
            self.decoder = shared_decoder(input_dims=64, output_dims=27)
        else:
            ValueError("Undefined encoder name: should be either 'current' or 'environment'.")

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent.view(latent.size(0), -1)
        out = self.decoder(latent)
        return out


class autoencoder():
    def __init__(self, encoder_name, lr, device, batch_size, after_epoch_callback=None):
        super(autoencoder, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.net = model(encoder_name).to(self.device)
        self.after_epoch_callback = after_epoch_callback
        self.loss_func = self.rmse_loss
        if encoder_name == 'current':
            self.stop_threshold = 1e-3
        elif encoder_name == 'environment':
            self.stop_threshold = 1e-4

    def rmse_loss(self, input, target):
        return torch.sqrt(((input - target) ** 2).mean())

    def fit(self, train_data, n_epochs=100, scheduler='constant', verbose=0):
        self.net.train()
        # define a progress bar
        if verbose:
            progress_bar = tqdm(range(n_epochs), desc=f'Epoch', ascii=True, dynamic_ncols=False)
        else:
            progress_bar = range(n_epochs)
        
        # define optimizer
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)

        # define training and validation data
        if scheduler == 'reduced':
            train_val_data = train_data
            # randomly split the training data into training and validation sets
            val_indices = np.random.choice(len(train_val_data), int(len(train_val_data)*0.2), replace=False)
            train_indices = np.setdiff1d(np.arange(len(train_val_data)), val_indices)
            train_data, val_data = train_val_data[train_indices].copy(), train_val_data[val_indices].copy()
            del train_val_data
            val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float())
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
            
            # define scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.6, patience=4, cooldown=6,
                threshold=1e-3, threshold_mode='rel', min_lr=self.lr*0.6**15
                )

            val_loss_log = np.zeros((n_epochs+4, len(val_loader))) * np.nan
            val_loss_log[:4,:] = np.array([[init_loss]*len(val_loader) for init_loss in range(100, 96, -1)])
        else:
            ValueError("Undefined scheduler: should be either 'constant' or 'reduced'.")

        # create training dataset, dataloader, and loss log
        train_dataset = datautils.custom_dataset(torch.from_numpy(train_data).float())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        loss_log = np.zeros((n_epochs, len(train_loader))) * np.nan

        # training loop
        self.epoch_n = 0
        self.iter_n = 0
        continue_training = True
        while continue_training:
            for train_batch_iter, (x, idx) in enumerate(train_loader, start=1):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                loss = self.loss_func(x, self.net(x))

                # backpropagation
                loss.backward()
                self.optimizer.step()

                # save iteration loss
                loss_log[self.epoch_n, train_batch_iter-1] = loss.item()
                self.iter_n += 1

            # if the scheduler is set to 'reduced', evaluate validation loss and update learning rate
            if scheduler == 'reduced':
                self.net.eval()
                with torch.no_grad():
                    for val_batch_iter, (x, idx) in enumerate(val_loader):
                        x = x.to(self.device)
                        val_loss = self.loss_func(x, self.net(x))
                        val_loss_log[self.epoch_n+4, val_batch_iter] = val_loss.item()
                self.scheduler.step(val_loss_log[self.epoch_n+4].mean())
                self.net.train()

                stop_condition1 = np.diff(val_loss_log[self.epoch_n:self.epoch_n+5,:].mean(axis=1))
                stop_condition1 = np.all(abs(stop_condition1/val_loss_log[self.epoch_n,:].mean())<self.stop_threshold)
                stop_condition2 = np.diff(val_loss_log[self.epoch_n:self.epoch_n+4,:].mean(axis=1))
                stop_condition2 = np.all(abs(stop_condition2/val_loss_log[self.epoch_n,:].mean())<self.stop_threshold/10)
                if stop_condition1 or stop_condition2:
                    # early stopping if validation loss converges
                    Warning('Early stopping due to validation loss convergence.')
                    break

            # save model if callback every several epochs
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self)

            # update progress bar if verbose
            if verbose:
                avg_batch_loss = loss_log[self.epoch_n, :].mean()
                if scheduler == 'reduced':
                    avg_val_loss = val_loss_log[self.epoch_n+4, :].mean()
                    current_lr = self.optimizer.param_groups[0]['lr']
                if verbose > 6:
                    if scheduler == 'reduced':
                        progress_bar.set_postfix(loss=avg_batch_loss, val_loss=avg_val_loss, lr=current_lr, refresh=False)
                    else:
                        progress_bar.set_postfix(loss=avg_batch_loss, refresh=False)
                    progress_bar.update(1)
                else: # update every 20% of the total epochs
                    step = n_epochs // (1+verbose*4)
                    if (self.epoch_n+1) % step == 0:
                        if scheduler == 'reduced':
                            progress_bar.set_postfix(loss=avg_batch_loss, val_loss=avg_val_loss, lr=current_lr, refresh=False)
                        else:
                            progress_bar.set_postfix(loss=avg_batch_loss, refresh=False)
                        progress_bar.update(step)

            self.epoch_n += 1
            if self.epoch_n >= n_epochs:
                continue_training = False
                break
        
        progress_bar.close()
        if self.after_epoch_callback is not None:
            self.after_epoch_callback(self, finish=True)

        return loss_log[~np.all(np.isnan(loss_log), axis=1)]

    def compute_loss(self, val_data):
        self.net.eval()
        val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float())
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        loss_log = np.zeros(len(val_loader)) * np.nan
        with torch.no_grad():
            for val_batch_iter, (x, idx) in enumerate(val_loader):
                x = x.to(self.device)
                loss_log[val_batch_iter] = self.loss_func(x, self.net(x)).item()
        return loss_log.mean()
    
    def encode(self, val_data, batch_size=512, encoding_window='full_series'):
        self.net.eval()
        if isinstance(val_data, torch.Tensor):
            dataset = datautils.custom_dataset(val_data)
        else:
            dataset = datautils.custom_dataset(torch.from_numpy(val_data).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        with torch.no_grad():
            encoded_data = []
            for x, _ in dataloader:
                x = x.to(self.device)
                out = self.net.encoder(x)
                encoded_data.append(out)
            encoded_data = torch.cat(encoded_data, dim=0)
        return encoded_data

    def save(self, fn):
        """
        Separately save the state dictionaries of the encoder and decoder.
        """
        torch.save(self.net.encoder.state_dict(), fn+'_encoder.pth')
        torch.save(self.net.decoder.state_dict(), fn+'_decoder.pth')


    def load(self, fn):
        """
        Load the model state and associated parameters from the specified file.
        """
        state_dict = torch.load(fn+'_encoder.pth', map_location=self.device, weights_only=True)
        self.net.encoder.load_state_dict(state_dict)
        state_dict = torch.load(fn+'_decoder.pth', map_location=self.device, weights_only=True)
        self.net.decoder.load_state_dict(state_dict)

        self.net.to(self.device)
        self.net.eval()

