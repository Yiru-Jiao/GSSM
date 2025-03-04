'''
This script defines the training, validation, and testing procedures for the posterior inference model.
'''

import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils.utils_data import DataOrganiser
from model import UnifiedProximity, LogNormalNLL, SmoothLogNormalNLL
from torch.utils.data import DataLoader


def set_experiments(stage=[1,2,3,4,5]):
    exp_config = []
    if 1 in stage: # single dataset, current only, no encoder pretraining
        exp_config.extend([
            [['highD'], ['current'], False],
            [['INTERACTION'], ['current'], False],
            [['SafeBaseline'], ['current'], False],
            [['Argoverse'], ['current'], False],
        ])
    if 2 in stage: # single dataset, current only, encoder pretrained with single dataset
        exp_config.extend([
            [['highD'], ['current'], True],
            [['INTERACTION'], ['current'], True],
            [['SafeBaseline'], ['current'], True],
            [['Argoverse'], ['current'], True],
        ])
    if 3 in stage: # multiple datasets, current only
        exp_config.extend([
            [['INTERACTION','highD'], ['current'], False],
            [['INTERACTION','highD','Argoverse'], ['current'], False],
            [['INTERACTION','highD','Argoverse','SafeBaseline'], ['current'], False],
            [['INTERACTION','highD'], ['current'], True],
            [['INTERACTION','highD','Argoverse'], ['current'], True],
        ])
    if 4 in stage: # single/multiple dataset, encoder pretrained with all datasets
        exp_config.extend([
            [['highD'], ['current'], 'all'],
            [['INTERACTION'], ['current'], 'all'],
            [['SafeBaseline'], ['current'], 'all'],
            [['Argoverse'], ['current'], 'all'],
            [['INTERACTION','highD'], ['current'], 'all'],
            [['INTERACTION','highD','Argoverse'], ['current'], 'all'],
            [['INTERACTION','highD','Argoverse','SafeBaseline'], ['current'], 'all'],
        ])
    if 5 in stage: # add extra features
        exp_config.extend([
            # [['Argoverse'], ['current','profiles'], False],
            # [['Argoverse'], ['current+acc','profiles'], False],
            # [['INTERACTION'], ['current','profiles'], False],
            # [['INTERACTION'], ['current+acc','profiles'], False],
            # [['highD'], ['current','profiles'], False],
            # [['highD'], ['current+acc','profiles'], False],
            # [['SafeBaseline'], ['current+acc'], True],
            # [['SafeBaseline'], ['current+acc', 'environment'], True],
            [['SafeBaseline'], ['current+acc','environment','profiles'], True],
            [['SafeBaseline'], ['current+acc'], 'all'],
            [['SafeBaseline'], ['current+acc', 'environment'], 'all'],
            [['SafeBaseline'], ['current+acc','environment','profiles'], 'all'],
        ])
    return exp_config


class train_val_test():
    def __init__(self, device, path_prepared, dataset,
                 encoder_selection='all', 
                 pretrained_encoder=False,
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
            if pretrained_encoder==False:
                self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/not_pretrained/{encoder_name}/'
            elif pretrained_encoder==True:
                self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/pretrained/{encoder_name}/'
            elif pretrained_encoder=='all':
                self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/pretrained_all/{encoder_name}/'
            os.makedirs(self.path_output, exist_ok=True)
        self.encoder_selection = encoder_selection
        self.pretrained_encoder = pretrained_encoder
        self.return_attention = return_attention
        self.define_model()

    def define_model(self,):
        self.model = UnifiedProximity(self.device, self.encoder_selection, self.return_attention)
        if self.pretrained_encoder==True:
            self.model.load_pretrained_encoders(self.dataset_name, self.path_prepared, continue_training=False)
        elif self.pretrained_encoder=='all':
            self.model.load_pretrained_encoders('INTERACTION_highD_Argoverse_SafeBaseline', 
                                                self.path_prepared, continue_training=True)

    def create_dataloader(self, batch_size):
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(DataOrganiser('train', self.dataset, self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(DataOrganiser('val', self.dataset, self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=False)
        
    def send_x_to_device(self, x):
        if isinstance(x, list):
            return tuple([i.to(self.device) for i in x])
        else:
            return x.to(self.device)
        
    def generate_noised_x(self, x, noise=0.01):
        '''
        Generate noise based on the range of each feature, and add it to the original features.
        Ensure the noised values of [l_ego, l_sur, combined_width, vy_ego, v_ego2, v_sur2, delta_v2]
        in current features and [v_ego] in profiles features are positive.
        '''
        if len(x.size())==2:
            if x.size(1)==12:
                # l_ego, l_sur, combined_width, vy_ego, vx_sur, vy_sur, v_ego2, v_sur2, delta_v2, delta_v, psi_sur, rho
                noise_ranges = torch.tensor([6., 6., 2., 15., 15., 15., 300., 300., 300., 15., np.pi, np.pi], requires_grad=False)
            elif x.size(1)==13:
                # l_ego, l_sur, combined_width, vy_ego, vx_sur, vy_sur, v_ego2, v_sur2, delta_v2, delta_v, psi_sur, acc_ego, rho
                noise_ranges = torch.tensor([6., 6., 2., 15., 15., 15., 300., 300., 300., 15., np.pi, 3.5, np.pi], requires_grad=False)
            noise = noise * noise_ranges.unsqueeze(0) * torch.randn_like(x, requires_grad=False)
            noised_x = x + noise
            noised_x.requires_grad = False
            # make sure the rad angles are within [-pi, pi]
            if x.size(1)==12:
                noised_x[:,[10,11]] = (noised_x[:,[10,11]] + np.pi) % (2 * np.pi) - np.pi
            elif x.size(1)==13:
                noised_x[:,[10,12]] = (noised_x[:,[10,12]] + np.pi) % (2 * np.pi) - np.pi
            # make sure some features are positive
            abs_mask = (noised_x>0).float() - (noised_x < 0).float()
            abs_mask.requires_grad = False
            abs_mask[:,[4,5]+list(range(9,x.size(1)))] = 1. # leave out vx_sur, vy_sur, delta_v, psi_sur, (acc_ego), rho
            noised_x = noised_x * abs_mask
        elif len(x.size())==3:
            noise_ranges = torch.tensor([3.5, 15., 15., 15], requires_grad=False)
            noise = noise * noise_ranges.unsqueeze(0).unsqueeze(0) * torch.randn_like(x, requires_grad=False)
            noised_x = x + noise
            abs_mask = (noised_x>0).float() - (noised_x < 0).float()
            abs_mask.requires_grad = False
            abs_mask[:,:,[0,2,3]] = 1. # leave out acc_ego, vx_sur, vy_sur
            noised_x = noised_x * abs_mask
        noised_x.requires_grad = x.requires_grad
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
        inducing_out = self.model(inducing_points)
        return inducing_out

    def compute_loss(self, x, y, return_out=False):
        if not self.lr_reduced:
            x = self.send_x_to_device(x)
            y = y.to(self.device)
            out = self.model(x)
            loss = self.loss_func(out, y)
        else:
            inducing_out = self.get_inducing_out(x)
            x = self.send_x_to_device(x)
            y = y.to(self.device)
            out = self.model(x)
            loss = self.later_loss_func(out, y, inducing_out)
        if return_out:
            return loss, out
        else:
            return loss

    def train_model(self, num_epochs=300, initial_lr=0.001, lr_schedule=True, verbose=0):
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.lr_reduced = False

        # Move model and loss function to device
        self.model = self.model.to(self.device)
        self.loss_func = LogNormalNLL().to(self.device)

        # Training
        loss_log = np.ones(num_epochs)*np.inf
        val_loss_log = np.ones(num_epochs)*np.inf

        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr)

        if lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.6, patience=10, cooldown=15,
                threshold=1e-3, threshold_mode='rel', verbose='deprecated', min_lr=self.initial_lr*0.6**15
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
                loss = self.compute_loss(x, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss
            loss_log[epoch_n] = train_loss.item() / train_batch_iter

            val_loss = self.val_loop()
            if lr_schedule and epoch_n>20: # Start learning rate scheduler after 20 epochs
                self.scheduler.step(val_loss)
                if not self.lr_reduced and self.optimizer.param_groups[0]['lr'] < self.initial_lr*0.5:
                    # we use self.initial_lr*0.5 rather than 0.6 to avoid missing due to float precision
                    sys.stderr.write('\n\n Learning rate is reduced twice so the loss will involve KL divergence since now...\n')
                    # re-define learning rate and its scheduler for new loss function
                    beta = 10.
                    self.later_loss_func = SmoothLogNormalNLL(beta).to(self.device)
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr)
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='min', factor=0.6, patience=5, cooldown=10,
                        threshold=1e-3, threshold_mode='rel', verbose='deprecated', min_lr=self.initial_lr*0.6**15
                    )
                    self.lr_reduced = True
            val_loss_log[epoch_n] = val_loss

            # Add information to progress bar with learning rate and loss values
            if self.verbose > 0:
                progress_bar.set_postfix(lr=self.optimizer.param_groups[0]['lr'],
                                         train_loss=loss_log[epoch_n], val_loss=val_loss, refresh=False)
                if epoch_n % self.verbose < 1:
                    progress_bar.update(self.verbose)

            # Early stopping if validation loss converges
            if (epoch_n>60) and np.all(abs(np.diff(val_loss_log[epoch_n-3:epoch_n+1])/val_loss_log[epoch_n-3:epoch_n])<5e-4):
                print(f'Validation loss converges and training stops early at Epoch {epoch_n}.')
                break

        progress_bar.close()

        # Print inspection results
        self.model.eval()

        mu_list, sigma_list = [], []
        val_gau = torch.tensor(0., device=self.device, requires_grad=False)
        val_smooth_gau = torch.tensor(0., device=self.device, requires_grad=False)
        with torch.no_grad():
            for val_batch_iter, (x, y) in enumerate(self.val_dataloader, start=1):
                self.lr_reduced = False
                gau_loss, out = self.compute_loss(x, y, return_out=True)
                self.lr_reduced = True
                smooth_gau_loss = self.compute_loss(x, y)
                mu, log_var = out[0], out[1] # mu: [batch_size], log_var: [batch_size]
                mu_list.append(mu.cpu().numpy())
                sigma_list.append(np.exp(0.5*log_var.cpu().numpy()))
                val_gau += gau_loss
                val_smooth_gau += smooth_gau_loss

        self.model.train()
        mu_sigma = pd.DataFrame(data={'mu': np.concatenate(mu_list), 'sigma': np.concatenate(sigma_list)})
        mu_sigma['mode'] = np.exp(mu_sigma['mu'] - mu_sigma['sigma']**2)
        print(mu_sigma.describe().to_string())
        print(f'Gaussian NLL: {val_gau.item()/val_batch_iter}, Smooth Gaussian NLL: {val_smooth_gau.item()/val_batch_iter}')

        # Save model and loss records
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
        self.model.eval()
        val_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        with torch.no_grad():
            for val_batch_iter, (x, y) in enumerate(self.val_dataloader, start=1):
                val_loss += self.compute_loss(x, y)
        self.model.train()
        return val_loss.item() / val_batch_iter
    
    def load_model(self, batch_size=None, initial_lr=None):
        if 'path_output' not in self.__dict__:
            if self.pretrained_encoder==False:
                self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/not_pretrained/{self.encoder_name}/'
            elif self.pretrained_encoder==True:
                self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/pretrained/{self.encoder_name}/'
            elif self.pretrained_encoder=='all':
                self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/pretrained_all/{self.encoder_name}/'
        if batch_size is not None and initial_lr is not None:
            self.path_save = self.path_output + f'bs={batch_size}-initlr={initial_lr}/'
        else:
            self.path_save = self.path_output
        final_model = glob.glob(self.path_save+'model_final*')[0]
        self.model.load_state_dict(torch.load(final_model, map_location=torch.device(self.device), weights_only=True))        
        self.model = self.model.to(self.device)
        self.loss_func = LogNormalNLL().to(self.device)
        self.model.eval()
