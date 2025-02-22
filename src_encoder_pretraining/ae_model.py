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
from src_encoder_pretraining.modules.regularizers import *


class shared_decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(shared_decoder, self).__init__()
        self.feature_extractor = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.feature_extractor(x)


class model(nn.Module):
    def __init__(self, encoder_name):
        super(model, self).__init__()
        if encoder_name == 'current':
            self.encoder = current_encoder(input_dims=1, output_dims=256)
            self.decoder = shared_decoder(input_dims=12*256, output_dims=12)
        elif encoder_name == 'current+acc':
            self.encoder = current_encoder(input_dims=1, output_dims=256)
            self.decoder = shared_decoder(input_dims=13*256, output_dims=13)
        elif encoder_name == 'environment':
            self.encoder = environment_encoder(input_dims=27, output_dims=256)
            self.decoder = shared_decoder(input_dims=256, output_dims=27)
        else:
            ValueError("Undefined encoder name: should be among 'current', 'current+acc', 'environment'.")

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent.reshape(latent.size(0), -1)
        out = self.decoder(latent)
        return out


class autoencoder():
    def __init__(self, encoder_name, lr, device, batch_size, after_epoch_callback=None):
        super(autoencoder, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.net = model(encoder_name).to(self.device)
        self.loss_log_vars = torch.nn.Parameter(torch.zeros(2, device=self.device))
        self.after_epoch_callback = after_epoch_callback
        if 'current' in encoder_name:
            self.stop_threshold = 1e-3
        elif encoder_name == 'environment':
            self.stop_threshold = 1e-4
        self.encoder_name = encoder_name


    # define eval() and train() functions
    def eval(self,):
        self.net.eval()
        self.loss_log_vars.requires_grad = False

    def train(self,):
        self.net.train()
        self.loss_log_vars.requires_grad = True


    def loss_func_topo(self, input, target): # used for environment encoder
        loss_ae = torch.sqrt(((input - target) ** 2).mean())
        loss_ae = 0.5 * torch.exp(-self.loss_log_vars[0]) * loss_ae*(1-torch.exp(-loss_ae)) + 0.5 * self.loss_log_vars[0]
        loss_topo = topo_loss(self, input)
        loss_topo = 0.5 * torch.exp(-self.loss_log_vars[1]) * loss_topo*(1-torch.exp(-loss_topo)) + 0.5 * self.loss_log_vars[1]
        return loss_ae + loss_topo
        

    def loss_func_ggeo(self, input, target): # used for current encoder
        loss_ae = torch.sqrt(((input - target) ** 2).mean())
        loss_ae = 0.5 * torch.exp(-self.loss_log_vars[0]) * loss_ae*(1-torch.exp(-loss_ae)) + 0.5 * self.loss_log_vars[0]
        loss_ggeo = geo_loss(self, input, 0.25)
        loss_ggeo = 0.5 * torch.exp(-self.loss_log_vars[1]) * loss_ggeo*(1-torch.exp(-loss_ggeo)) + 0.5 * self.loss_log_vars[1]
        return loss_ae + loss_ggeo


    def fit(self, train_data, n_epochs=100, scheduler='constant', verbose=0):
        self.train()
        # define a progress bar
        if verbose:
            progress_bar = tqdm(range(n_epochs), desc=f'Epoch', ascii=True, dynamic_ncols=False)
        else:
            progress_bar = range(n_epochs)
        
        # define optimizer for the net and log_vars
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        self.optimizer.add_param_group({'params': self.loss_log_vars, 'lr': self.lr})

        # define training and validation data
        if scheduler == 'reduced':
            train_val_data = train_data
            # randomly split the training data into training and validation sets
            val_indices = np.random.choice(len(train_val_data), int(len(train_val_data)*0.2), replace=False)
            train_indices = np.setdiff1d(np.arange(len(train_val_data)), val_indices)
            train_data, val_data = train_val_data[train_indices].copy(), train_val_data[val_indices].copy()
            del train_val_data
            val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float())
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
            
            # define scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.6, patience=4, cooldown=6,
                threshold=1e-3, threshold_mode='rel', min_lr=self.lr*0.6**15
                )
            val_loss_log = np.zeros(n_epochs) * np.nan

        # create training dataset, dataloader, and loss log
        train_dataset = datautils.custom_dataset(torch.from_numpy(train_data).float())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        loss_log = np.zeros(n_epochs) * np.nan

        # training loop
        self.epoch_n = 0
        self.iter_n = 0
        continue_training = True
        while continue_training:
            train_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
            for train_batch_iter, (x, idx) in enumerate(train_loader, start=1):
                self.optimizer.zero_grad()
                if 'current' in self.encoder_name:
                    x = x.to(self.device)
                    loss = self.loss_func_ggeo(x, self.net(x))
                elif self.encoder_name == 'environment':
                    x = x.to(self.device)
                    loss = self.loss_func_topo(x, self.net(x))
                loss.backward()
                self.optimizer.step()
                train_loss += loss
                self.iter_n += 1

            loss_log[self.epoch_n] = train_loss.item() / train_batch_iter

            # if the scheduler is set to 'reduced', evaluate validation loss and update learning rate
            if scheduler == 'reduced':
                self.eval()
                with torch.no_grad():
                    val_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                    for val_batch_iter, (x, idx) in enumerate(val_loader, start=1):
                        if 'current' in self.encoder_name:
                            x = x.to(self.device)
                            val_loss += self.loss_func_ggeo(x, self.net(x))
                        elif self.encoder_name == 'environment':
                            x = x.to(self.device)
                            val_loss += self.loss_func_topo(x, self.net(x))
                    val_loss_log[self.epoch_n] = val_loss.item() / val_batch_iter
                self.train()
                if self.epoch_n >= 20: # start scheduler after 20 epochs
                    self.scheduler.step(val_loss_log[self.epoch_n])
                    stop_condition = np.diff(val_loss_log[self.epoch_n-3:self.epoch_n+1]) # diff in the last 3 epochs
                    stop_condition = np.all(abs(stop_condition/val_loss_log[self.epoch_n-3:self.epoch_n])<self.stop_threshold)
                    if stop_condition:
                        # early stopping if validation loss converges
                        print('Early stopping due to validation loss convergence.')
                        continue_training = False

            # save model if callback every several epochs
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self)

            # update progress bar if verbose
            if verbose > 6:
                if scheduler == 'reduced':
                    progress_bar.set_postfix(loss=loss_log[self.epoch_n], 
                                             val_loss=val_loss_log[self.epoch_n], 
                                             lr=self.optimizer.param_groups[0]['lr'], refresh=False)
                else:
                    progress_bar.set_postfix(loss=loss_log[self.epoch_n], refresh=False)
                progress_bar.update(1)
            else:
                step = n_epochs // (1+verbose*4)
                if (self.epoch_n+1) % step == 0:
                    if scheduler == 'reduced':
                        progress_bar.set_postfix(loss=loss_log[self.epoch_n], 
                                                 val_loss=val_loss_log[self.epoch_n],
                                                 lr=self.optimizer.param_groups[0]['lr'], refresh=False)
                    else:
                        progress_bar.set_postfix(loss=loss_log[self.epoch_n], refresh=False)
                    progress_bar.update(step)

            self.epoch_n += 1
            if self.epoch_n >= n_epochs:
                continue_training = False
                break
        
        if verbose:
            progress_bar.close()
        if self.after_epoch_callback is not None:
            self.after_epoch_callback(self, finish=True)

        return loss_log[:self.epoch_n+1], val_loss_log[:self.epoch_n+1] if scheduler == 'reduced' else np.zeros_like(loss_log)*np.nan


    def compute_loss(self, val_data):
        org_training = self.net.training
        self.eval()

        val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float())
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        with torch.no_grad():
            for val_batch_iter, (x, idx) in enumerate(val_loader, start=1):
                if 'current' in self.encoder_name:
                    x = x.to(self.device)
                    loss += self.loss_func_ggeo(x, self.net(x))
                elif self.encoder_name == 'environment':
                    x = x.to(self.device)
                    loss += self.loss_func_topo(x, self.net(x))

        if org_training:
            self.train()
        return loss.item() / val_batch_iter
    

    def encode(self, val_data, batch_size=512):
        org_training = self.net.training
        self.eval()

        if not isinstance(val_data, torch.Tensor):
            val_data = torch.from_numpy(val_data).float()
        
        if val_data.size(0) < batch_size:
            with torch.no_grad():
                encoded_data = self.net.encoder(val_data) # val_data is already on the device
        else:
            dataset = datautils.custom_dataset(val_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            with torch.no_grad():
                encoded_data = []
                for x, _ in dataloader:
                    x = x.to(self.device)
                    out = self.net.encoder(x)
                    encoded_data.append(out)
                encoded_data = torch.cat(encoded_data, dim=0)
        
        if org_training:
            self.train()
        return encoded_data # (n_samples, 12, 256) for current, (n_samples, 13, 256) for current+acc, (n_samples, 1, 256) for environment


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

