'''
This script defines the training, validation, and testing procedures for the posterior inference model.
'''

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils.utils_data import DataOrganiser
from model import UnifiedProximity, LogNormalNLL, SmoothLogNormalNLL
from torch.utils.data import DataLoader


def set_experiments(stage=[1,2,3]):
    exp_config = []
    if 1 in stage: # single dataset, current only
        exp_config.extend([
            [['SafeBaseline'], ['current']],
            [['highD'], ['current']],
            [['ArgoverseHV'], ['current']],
        ])
    if 2 in stage: # multiple datasets, current only
        exp_config.extend([
            [['SafeBaseline','ArgoverseHV'], ['current']],
            [['SafeBaseline','highD'], ['current']],
            # [['SafeBaseline','ArgoverseHV','highD'], ['current']],
        ])
    if 3 in stage: # add extra features
        exp_config.extend([
            [['SafeBaseline'], ['current+acc']],
            [['SafeBaseline'], ['current', 'environment']],
            [['SafeBaseline'], ['current+acc', 'environment']],
            [['SafeBaseline'], ['current','environment','profiles']],
            [['SafeBaseline'], ['current+acc','environment','profiles']],
        ])
    return exp_config


class train_val_test():
    def __init__(self, device, path_prepared, dataset,
                 encoder_selection='all', 
                 single_output=None,
                 return_attention=False):
        super(train_val_test, self).__init__()
        self.device = device
        self.path_prepared = path_prepared
        self.dataset = dataset
        dataset_name = '_'.join(dataset)
        self.dataset_name = dataset_name
        if encoder_selection == 'all':
            encoder_selection = ['current+acc', 'environment', 'profiles']
        encoder_name = '_'.join(encoder_selection)
        self.encoder_name = encoder_name
        if not return_attention:
            self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/{encoder_name}/'
            os.makedirs(self.path_output, exist_ok=True)
        self.encoder_selection = encoder_selection
        self.single_output = single_output
        self.return_attention = return_attention
        self._model = UnifiedProximity(self.encoder_selection, self.single_output, self.return_attention)

    def create_dataloader(self, batch_size, mixrate=2, random_seed=131):
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(DataOrganiser('train', self.dataset, self.encoder_selection, self.path_prepared, 
                                                         mixrate, random_seed), batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(DataOrganiser('val', self.dataset, self.encoder_selection, self.path_prepared, 
                                                       mixrate, random_seed), batch_size=self.batch_size, shuffle=False)
        self.current_ranges = self.train_dataloader.dataset.data[0].var(dim=0).sqrt()
        if 'profiles' in self.encoder_selection:
            self.profile_ranges = self.train_dataloader.dataset.data[-2].reshape(-1, 4).var(dim=0).sqrt()
        
    def send_x_to_device(self, x):
        if isinstance(x, list):
            return tuple([i.to(self.device) for i in x])
        else:
            return x.to(self.device)
        
    def generate_noised_x(self, x, noise=0.01):
        '''
        Generate noise based on the range of each feature, and add it to the original features.
        Maintain the original values of [vy_ego, v_ego2, v_sur2, delta_v2].
        '''
        if len(x.size())==2:
            if self._model.training:
                noise = noise * self.current_ranges.unsqueeze(0) * torch.randn_like(x, requires_grad=False)
            else: # generate fixed noise for validation and testing
                noise_mask = torch.ones_like(x, requires_grad=False)
                noise_mask[::2, :] *= -1
                noise_mask[:, ::2] *= -1
                noise = noise * self.current_ranges.unsqueeze(0) * noise_mask
            noised_x = x + noise
            # make sure the rad angles are within [-pi, pi]
            if x.size(1)==12:
                noised_x[:,[10,11]] = (noised_x[:,[10,11]] + np.pi) % (2 * np.pi) - np.pi
            elif x.size(1)==13:
                noised_x[:,[10,12]] = (noised_x[:,[10,12]] + np.pi) % (2 * np.pi) - np.pi
        elif len(x.size())==3:
            if self._model.training:
                noise = noise * self.profile_ranges.unsqueeze(0).unsqueeze(0) * torch.randn_like(x, requires_grad=False)
            else: # generate fixed noise for validation and testing
                noise_mask = torch.ones_like(x, requires_grad=False)
                noise_mask[::2, :, :] *= -1
                noise_mask[:, ::2, :] *= -1
                noise_mask[:, :, ::2] *= -1
                noise = noise * self.profile_ranges.unsqueeze(0).unsqueeze(0) * noise_mask
            noised_x = x + noise
        return noised_x

    def get_inducing_out(self, x):
        if self.encoder_selection==['current'] or self.encoder_selection==['current+acc']:
            inducing_points = self.generate_noised_x(x)
        elif self.encoder_selection==['current','environment'] or self.encoder_selection==['current+acc','environment']:
            inducing_points = [self.generate_noised_x(x[0]), x[1]]
        elif self.encoder_selection==['current','profiles'] or self.encoder_selection==['current+acc','profiles']:
            inducing_points = [self.generate_noised_x(x[0]), self.generate_noised_x(x[1])]
        elif self.encoder_selection==['current','environment','profiles'] or self.encoder_selection==['current+acc','environment','profiles']:
            inducing_points = [self.generate_noised_x(x[0]), x[1], self.generate_noised_x(x[2])]
        inducing_points = self.send_x_to_device(inducing_points)
        if self._model.training:
            inducing_out = self._model(inducing_points)
        else:
            with torch.no_grad():
                inducing_out = self._model(inducing_points)
        return inducing_out

    def compute_loss(self, x, y, model2use, return_out=False, smoothed=True):
        if not smoothed:
            x = self.send_x_to_device(x)
            y = y.to(self.device)
            out = model2use(x)
            loss = self.lognorm_nll(out, y)
        else:
            inducing_out = self.get_inducing_out(x)
            x = self.send_x_to_device(x)
            y = y.to(self.device)
            out = model2use(x)
            loss = self.smooth_lognorm_nll(out, y, inducing_out)
        if return_out:
            return loss, out
        else:
            return loss

    def train_model(self, num_epochs=100, initial_lr=0.0001, lr_schedule=True, verbose=0):
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.lr_reduced = False
        self.epoch_reduced = num_epochs
        # Move model and loss function to device
        self._model = self._model.to(self.device)
        self.lognorm_nll = LogNormalNLL().to(self.device)
        self.smooth_lognorm_nll = SmoothLogNormalNLL(beta=5).to(self.device)

        # Training
        loss_log = np.ones(num_epochs)*np.inf
        val_loss_log = np.ones(num_epochs)*np.inf

        self._model.train()
        self.optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.initial_lr)

        if lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.6, patience=5, cooldown=0,
                threshold=1e-3, threshold_mode='rel', verbose='deprecated', min_lr=self.initial_lr*0.6**30
            )

        if self.verbose > 0:
            progress_bar = tqdm(range(num_epochs), unit='epoch', ascii=True, dynamic_ncols=False)
        else:
            progress_bar = range(num_epochs)

        for epoch_n in progress_bar:
            train_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
            for train_batch_iter, (x, y) in enumerate(self.train_dataloader, start=1):
                '''
                x: tuple of features, each of shape [batch_size, feature_dims]
                y: [batch_size]
                '''
                self.optimizer.zero_grad()
                loss = self.compute_loss(x, y, self._model)
                loss.backward()
                self.optimizer.step()
                train_loss += loss
                if self.lr_reduced: # Update the averaged model after learning rate is reduced
                    self.model.update_parameters(self._model)
            loss_log[epoch_n] = train_loss.item() / train_batch_iter

            val_loss = self.val_loop()
            if lr_schedule and epoch_n>25: # Start learning rate scheduler after 25 epochs
                if not self.lr_reduced:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                if not self.lr_reduced and self.optimizer.param_groups[0]['lr'] < self.initial_lr*0.8:
                    # we use self.initial_lr*0.8 rather than 0.6 to avoid missing due to float precision
                    # set the flag to True to update the averaged model
                    self.lr_reduced = True
                    self.epoch_reduced = epoch_n
                    # use an averaged model for the rest of training
                    self.model = torch.optim.swa_utils.AveragedModel(self._model)
                    self.model.update_parameters(self._model)
                    self.model = self.model.to(self.device)
                    # print the current learning rate and validation loss
                    val_loss = self.val_loop()
                    sys.stderr.write('\n Learning rate is reduced so the training uses SWA since now...')
                    sys.stderr.write(f'\n Current lr: {self.optimizer.param_groups[0]["lr"]}, epoch: {epoch_n}, val_loss: {val_loss}')
                    # new scheduler
                    self.scheduler = torch.optim.swa_utils.SWALR(self.optimizer, 
                                                                 swa_lr=self.optimizer.param_groups[0]['lr'] * 0.05, # 3e-6
                                                                 anneal_epochs=20,
                                                                 anneal_strategy="cos")
                    self.model.train()
            val_loss_log[epoch_n] = val_loss

            # Add information to progress bar with learning rate and loss values
            if self.verbose == 1:
                progress_bar.set_postfix(lr=self.optimizer.param_groups[0]['lr'],
                                         train_loss=loss_log[epoch_n], val_loss=val_loss, refresh=False)
            elif self.verbose > 1:
                progress_bar.set_postfix(lr=self.optimizer.param_groups[0]['lr'],
                                         train_loss=loss_log[epoch_n], val_loss=val_loss, refresh=False)
                if epoch_n % self.verbose == (self.verbose-1):
                    progress_bar.update(self.verbose)

            # Early stopping if validation loss converges
            if (epoch_n>50) and np.all(abs(np.diff(val_loss_log[epoch_n-3:epoch_n+1])/val_loss_log[epoch_n-3:epoch_n])<1e-4):
                print(f'Validation loss converges and training stops early at Epoch {epoch_n}.')
                break

            # Force a stop if the swa procedure is long enough
            if epoch_n > (self.epoch_reduced+50) and (self.optimizer.param_groups[0]['lr'] < 5e-6):
                print(f'Learning procedure is too long and training stops early at Epoch {epoch_n}.')
                break

        progress_bar.close()

        # Save model and loss records
        self.customed_update_bn(self.train_dataloader, self.model)
        torch.save(self.model.state_dict(), self.path_output+f'model_final_{epoch_n}epoch.pth')
        loss_log = loss_log[:epoch_n+1]
        if lr_schedule:
            val_loss_log = val_loss_log[:epoch_n+1]
            loss_log = pd.DataFrame(index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                    data={'train_loss': loss_log, 'val_loss': val_loss_log})
            loss_log.to_csv(self.path_output+'loss_log.csv')
            self.val_loss_log = val_loss_log
        else:
            print('No learning rate scheduler has been used.')
            loss_log = pd.DataFrame(index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                    data={'train_loss': loss_log})
            loss_log.to_csv(self.path_output+'loss_log.csv')

    # Validation loop
    def val_loop(self):
        if self.lr_reduced:
            model2use = self.model
        else:
            model2use = self._model
        model2use.eval()
        val_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        with torch.no_grad():
            for val_batch_iter, (x, y) in enumerate(self.val_dataloader, start=1):
                val_loss += self.compute_loss(x, y, model2use)
        model2use.train()
        return val_loss.item() / val_batch_iter
    
    @torch.no_grad()
    def customed_update_bn(self, loader, model):
        '''
        Update BatchNorm running_mean, running_var buffers in the model.
        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.

        This function is a customed version of the original one in torch.optim.swa_utils.update_bn
        for the list format of input data.
        '''
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                momenta[module] = module.momentum
        if not momenta:
            return
        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
        for x, _ in loader:
            x = self.send_x_to_device(x)
            model(x)
        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)

    def load_model(self, mixrate=2):
        if 'path_output' not in self.__dict__:
            self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/{self.encoder_name}/'
        if mixrate<=1 and 'mixed' not in self.path_output:
            self.path_output = f'{self.path_output}mixed{mixrate}/'
        final_model = [f for f in os.listdir(self.path_output) if f.endswith('.pth')][0]
        final_model = os.path.join(self.path_output, final_model)
        self.model.load_state_dict(torch.load(final_model, map_location=torch.device(self.device), weights_only=True))
        self._model.load_state_dict(self.model.module.state_dict())
        self.model = self.model.to(self.device)
        self._model = self._model.to(self.device)
        self.model.eval()
        self._model.eval()
        self.lognorm_nll = LogNormalNLL().to(self.device)
        self.smooth_lognorm_nll = SmoothLogNormalNLL(beta=5).to(self.device)

    def print_inspection(self):
        self.model.eval()
        mu_list, sigma_list = [], []
        val_gau = torch.tensor(0., device=self.device, requires_grad=False)
        val_smooth_gau = torch.tensor(0., device=self.device, requires_grad=False)
        with torch.no_grad():
            for val_batch_iter, (x, y) in enumerate(self.val_dataloader, start=1):
                gau_loss, out = self.compute_loss(x, y, self.model, return_out=True, smoothed=False)
                smooth_gau_loss = self.compute_loss(x, y, self.model)
                mu, log_var = out[0], out[1] # mu: [batch_size], log_var: [batch_size]
                mu_list.append(mu.cpu().numpy())
                sigma_list.append(np.exp(0.5*log_var.cpu().numpy()))
                val_gau += gau_loss
                val_smooth_gau += smooth_gau_loss

        mu_sigma = pd.DataFrame(data={'mu': np.concatenate(mu_list), 'sigma': np.concatenate(sigma_list)})
        mu_sigma['mode'] = np.exp(mu_sigma['mu'] - mu_sigma['sigma']**2)
        print(mu_sigma.describe().to_string())
        print(f'LogNormal NLL: {val_gau.item()/val_batch_iter}, Smooth LogNormal NLL: {val_smooth_gau.item()/val_batch_iter}')

