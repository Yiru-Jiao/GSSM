'''
train_val_test: a class that handles the training and validation process of the SVGP model. 
                   It creates the dataloaders, defines the model, likelihood, and loss function, 
                   and performs the training loop.
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


class train_val_test():
    def __init__(self, device, path_prepared, 
                 encoder_selection='all', 
                 cross_attention='all', 
                 pretrained_encoder=False,
                 return_attention=False):
        super(train_val_test, self).__init__()
        self.device = device
        self.path_prepared = path_prepared
        if encoder_selection == 'all':
            encoder_selection = ['current', 'environment', 'profiles']
        encoder_name = '_'.join(encoder_selection)
        if cross_attention == 'all':
            cross_attention = ['first', 'middle', 'last']
        cross_attention_name = '_'.join(cross_attention) if len(cross_attention) > 0 else 'not_crossed'
        if not return_attention:
            if pretrained_encoder:
                self.path_output = path_prepared + f'PosteriorInference/pretrained/{encoder_name}_{cross_attention_name}/'
            else:
                self.path_output = path_prepared + f'PosteriorInference/notpretrained/{encoder_name}_{cross_attention_name}/'
            os.makedirs(self.path_output, exist_ok=True)
            self.loss_func = LogNormalNLL()
        self.encoder_selection = encoder_selection
        self.cross_attention = cross_attention
        self.pretrained_encoder = pretrained_encoder
        self.return_attention = return_attention
        self.define_model()

    def define_model(self,):
        self.model = UnifiedProximity(self.device, self.encoder_selection, self.cross_attention, self.return_attention)
        if self.pretrained_encoder:
            self.model.load_pretrained_encoders(self.path_prepared)

    def create_dataloader(self, batch_size):
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(DataOrganiser('train', self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(DataOrganiser('val', self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(DataOrganiser('test', self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=False)

    def send_x_to_device(self, x):
        if isinstance(x, list):
            return [i.to(self.device) for i in x]
        else:
            return x.to(self.device)

    def train_model(self, num_epochs=500, initial_lr=0.001, lr_schedule=True, verbose=0):
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.path_save = self.path_output + f'bs={self.batch_size}-initlr={self.initial_lr}/'
        os.makedirs(self.path_save, exist_ok=True)

        # Move model and loss function to device
        self.model = self.model.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        # Training
        num_batches = len(self.train_dataloader)
        loss_log = np.zeros((num_epochs, num_batches))
        val_loss_log = [100., 99., 98., 97., 96., 95.]

        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.initial_lr, amsgrad=True)

        if lr_schedule:
            if 'profiles' in self.encoder_selection:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.6, patience=4, cooldown=4,
                    threshold=1e-3, threshold_mode='rel', verbose='deprecated', min_lr=self.initial_lr*0.6**15
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.6, patience=4, cooldown=8,
                    threshold=1e-3, threshold_mode='rel', verbose='deprecated', min_lr=self.initial_lr*0.6**15
                )

        if self.verbose > 0:
            progress_bar = tqdm(range(num_epochs), unit='epoch', ascii=True, dynamic_ncols=False, miniters=self.verbose)
        else:
            progress_bar = range(num_epochs)
        for epoch_n in progress_bar:
            for train_batch_iter, (x, y) in enumerate(self.train_dataloader):
                out = self.model(self.send_x_to_device(x))
                loss = self.loss_func(out, y.to(self.device))
                loss_log[epoch_n, train_batch_iter] = loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if lr_schedule:
                val_loss = self.val_loop()
                self.scheduler.step(val_loss)
                val_loss_log.append(val_loss)

                # Add information to progress bar with learning rate and loss values
                progress_bar.set_postfix(lr=self.optimizer.param_groups[0]['lr'],
                                            train_loss=loss_log[epoch_n].mean(), val_loss=val_loss, refresh=False)
                if epoch_n % self.verbose == 0:
                    progress_bar.update(self.verbose)

            stop_condition1 = np.all(abs(np.diff(val_loss_log)[-5:]/val_loss_log[-5:])<1e-3)
            stop_condition2 = np.all(abs(np.diff(val_loss_log)[-4:]/val_loss_log[-4:])<1e-4)
            # Early stopping if validation loss converges
            if stop_condition1 or stop_condition2:
                print(f'Validation loss converges and training stops early at Epoch {epoch_n}.')
                break

        if lr_schedule:
            # Save model and loss records
            torch.save(self.model.state_dict(), self.path_save+f'model_final_{epoch_n}epoch.pth')
            loss_log = loss_log[loss_log.sum(axis=1)>0]
            loss_log = pd.DataFrame(loss_log, index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                    columns=[f'iter_{i}' for i in range(1, len(loss_log[0])+1)])
            loss_log.to_csv(self.path_save+'loss_log.csv')
            val_loss_log = pd.DataFrame(val_loss_log[5:], index=[f'epoch_{i}' for i in range(1, len(val_loss_log)-4)], columns=['val_loss'])
            val_loss_log.to_csv(self.path_save+'val_loss_log.csv')

    # Validation loop
    def val_loop(self,):
        self.model.eval()
        val_loss = 0.
        with torch.no_grad():
            for val_batch_iter, (x, y) in enumerate(self.val_dataloader):
                out = self.model(self.send_x_to_device(x))
                loss = self.loss_func(out, y.to(self.device)).item()
                val_loss += loss
        self.model.train()
        return val_loss/(val_batch_iter+1)
    
    def load_model(self, batch_size, initial_lr):
        self.path_save = self.path_output + f'bs={batch_size}-initlr={initial_lr}/'
        final_model = glob.glob(self.path_save+'model_final*')[0]
        self.model.load_state_dict(torch.load(final_model, map_location=torch.device(self.device), weights_only=True))        
        self.model = self.model.to(self.device)
        self.loss_func = self.loss_func.to(self.device)
        self.model.eval()

    def test_model(self, batch_size=None, initial_lr=None):
        if batch_size is not None and initial_lr is not None:
            # Load trained model and likelihood
            self.load_model(batch_size, initial_lr)

        # Evaluate the model
        self.model.eval()
        test_loss = np.zeros(len(self.test_dataloader))
        with torch.no_grad():
            for test_batch_iter, (x, y) in enumerate(self.test_dataloader):
                out = self.model(self.send_x_to_device(x))
                loss = self.loss_func(out, y.to(self.device)).item()
                test_loss[test_batch_iter] = loss
        self.model.train()
        return test_loss.mean()
