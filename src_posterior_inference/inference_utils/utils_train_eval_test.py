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
from model import UnifiedProximity, LogNormalNLL
from torch.utils.data import DataLoader


def set_experiments(stage=[1,2,3,4,5]):
    exp_config = []
    if 1 in stage: # pretraining encoder to maintain feature structure
        exp_config.extend([
            [['highD'], ['current'], False],
            [['highD'], ['current'], True],
            [['highD'], ['current','profiles'], False],
            [['highD'], ['current','profiles'], True],
        ])
    if 2 in stage: # single dataset, current only, encoder pretrained with single dataset
        exp_config.extend([
            [['SafeBaseline'], ['current'], False],
            [['INTERACTION'], ['current'], False],
            [['Argoverse'], ['current'], False],
            # [['SafeBaseline'], ['current'], True],
            # [['INTERACTION'], ['current'], True],
            # [['Argoverse'], ['current'], True],
        ])
    if 3 in stage: # multiple datasets, current only
        exp_config.extend([
            [['Argoverse', 'INTERACTION'], ['current'], False],
            [['Argoverse', 'INTERACTION', 'SafeBaseline'], ['current'], False],
            [['Argoverse', 'INTERACTION', 'SafeBaseline', 'highD'], ['current'], False],
            [['Argoverse', 'INTERACTION'], ['current'], True],
            [['Argoverse', 'INTERACTION', 'SafeBaseline'], ['current'], True],
            [['Argoverse', 'INTERACTION', 'SafeBaseline', 'highD'], ['current'], True],
        ])
    if 4 in stage: # single dataset, encoder pretrained with all datasets
        exp_config.extend([
            [['highD'], ['current'], True],
            [['SafeBaseline'], ['current'], True],
            [['INTERACTION'], ['current'], True],
            [['Argoverse'], ['current'], True],
        ])
    if 5 in stage: # on SafeBaseline, add extra features
        exp_config.extend([
            [['SafeBaseline'], ['current+acc'], False],
            [['SafeBaseline'], ['current+acc', 'environment'], False],
            [['SafeBaseline'], ['current+acc','environment','profiles'], False],
            [['SafeBaseline'], ['current+acc'], True],
            [['SafeBaseline'], ['current+acc', 'environment'], True],
            [['SafeBaseline'], ['current+acc','environment','profiles'], True],
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
            if pretrained_encoder:
                self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/pretrained/{encoder_name}/'
            else:
                self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/not_pretrained/{encoder_name}/'
            os.makedirs(self.path_output, exist_ok=True)
        self.encoder_selection = encoder_selection
        self.pretrained_encoder = pretrained_encoder
        self.return_attention = return_attention
        self.define_model()
        self.loss_func = LogNormalNLL()

    def define_model(self,):
        self.model = UnifiedProximity(self.device, self.encoder_selection, self.return_attention)
        if self.pretrained_encoder:
            self.model.load_pretrained_encoders(self.dataset_name, self.path_prepared, continue_training=False)

    def create_dataloader(self, batch_size):
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(DataOrganiser('train', self.dataset, self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(DataOrganiser('val', self.dataset, self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=False)
        
    def send_x_to_device(self, x):
        if isinstance(x, tuple):
            return tuple([i.to(self.device) for i in x])
        else:
            return x.to(self.device)

    def train_model(self, num_epochs=300, initial_lr=0.001, lr_schedule=True, verbose=0):
        self.initial_lr = initial_lr
        self.verbose = verbose

        # Move model and loss function to device
        self.model = self.model.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        # Training
        loss_log = np.ones(num_epochs)*np.inf
        val_loss_log = np.ones(num_epochs)*np.inf

        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr)

        if lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.6, patience=10, cooldown=20,
                threshold=1e-3, threshold_mode='rel', verbose='deprecated', min_lr=self.initial_lr*0.6**15
            )

        if self.verbose > 0:
            progress_bar = tqdm(range(num_epochs), unit='epoch', ascii=True, dynamic_ncols=False)
        else:
            progress_bar = range(num_epochs)

        if 'profiles' in self.encoder_selection:
            scaler = torch.amp.GradScaler()  # Initialize gradient scaler for mixed precision training
        for epoch_n in progress_bar:
            train_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
            for train_batch_iter, (x, y) in enumerate(self.train_dataloader, start=1):
                self.optimizer.zero_grad()
                if 'profiles' in self.encoder_selection:
                    with torch.amp.autocast(device_type="cuda"):  # Enables Mixed Precision
                        out = self.model(self.send_x_to_device(x))
                        loss = self.loss_func(out, y.to(self.device))
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    out = self.model(self.send_x_to_device(x))
                    loss = self.loss_func(out, y.to(self.device))
                    loss.backward()
                    self.optimizer.step()
                train_loss += loss
            loss_log[epoch_n] = train_loss.item() / train_batch_iter

            val_loss = self.val_loop()
            if lr_schedule:
                self.scheduler.step(val_loss)
            val_loss_log[epoch_n] = val_loss

            # Add information to progress bar with learning rate and loss values
            if self.verbose > 0:
                progress_bar.set_postfix(lr=self.optimizer.param_groups[0]['lr'],
                                         train_loss=loss_log[epoch_n], val_loss=val_loss, refresh=False)
                if epoch_n % self.verbose < 1:
                    progress_bar.update(self.verbose)

            # Early stopping if validation loss converges
            if (epoch_n>100) and np.all(abs(np.diff(val_loss_log[epoch_n-3:epoch_n+1])/val_loss_log[epoch_n-3:epoch_n])<5e-4):
                print(f'Validation loss converges and training stops early at Epoch {epoch_n}.')
                break

        progress_bar.close()
        if lr_schedule:
            # Save model and loss records
            torch.save(self.model.state_dict(), self.path_output+f'model_final_{epoch_n}epoch.pth')
            loss_log = loss_log[loss_log<np.inf]
            val_loss_log = val_loss_log[val_loss_log<np.inf]
            loss_log = pd.DataFrame(index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                    data={'train_loss': loss_log, 'val_loss': val_loss_log})
            loss_log.to_csv(self.path_output+'loss_log.csv')
            self.val_loss_log = val_loss_log
        else:
            print('No learning rate scheduler has been used.')

    # Validation loop
    def val_loop(self,):
        self.model.eval()
        val_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        with torch.no_grad():
            for val_batch_iter, (x, y) in enumerate(self.val_dataloader, start=1):
                if 'profiles' in self.encoder_selection:
                    with torch.amp.autocast(device_type="cuda"):  # Enables Mixed Precision
                        out = self.model(self.send_x_to_device(x))
                        val_loss += self.loss_func(out, y.to(self.device))
                else:
                    out = self.model(self.send_x_to_device(x))
                    val_loss += self.loss_func(out, y.to(self.device))
        self.model.train()
        return val_loss.item() / val_batch_iter
    
    def load_model(self, batch_size=None, initial_lr=None):
        if 'path_output' not in self.__dict__:
            if self.pretrained_encoder:
                self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/pretrained/{self.encoder_name}/'
            else:
                self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/not_pretrained/{self.encoder_name}/'
        if batch_size is not None and initial_lr is not None:
            self.path_save = self.path_output + f'bs={batch_size}-initlr={initial_lr}/'
        else:
            self.path_save = self.path_output
        final_model = glob.glob(self.path_save+'model_final*')[0]
        self.model.load_state_dict(torch.load(final_model, map_location=torch.device(self.device), weights_only=True))        
        self.model = self.model.to(self.device)
        self.loss_func = self.loss_func.to(self.device)
        self.model.eval()
