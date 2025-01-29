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


def set_experiments(stage=[1,2,3,4]):
    exp_config = []
    if 1 in stage:
        exp_config.extend([
            [['highD'], ['current'], [], False],
            [['SafeBaseline'], ['current'], [], False],
            [['SafeBaseline'], ['current', 'environment'], [], False],
            [['INTERACTION'], ['current'], [], False],
            [['Argoverse'], ['current'], [], False],
        ])
    if 2 in stage:
        exp_config.extend([
            [['highD'], ['current'], [], True],
            [['SafeBaseline'], ['current'], [], True],
            [['INTERACTION'], ['current'], [], True],
            [['Argoverse'], ['current'], [], True],
        ])
    if 3 in stage:
        exp_config.extend([
            [['Argoverse', 'SafeBaseline'], ['current'], [], False],
            [['Argoverse', 'SafeBaseline', 'INTERACTION'], ['current'], [], False],
            [['Argoverse', 'SafeBaseline', 'INTERACTION', 'highD'], ['current'], [], False],
        ])
    if 4 in stage:
        exp_config.extend([
            [['SafeBaseline'], ['current','environment','profiles'], [], False],
            # [['SafeBaseline'], ['current','environment','profiles'], [], True],
        ])
            # [[], ['current','profiles'], [], False],
            # [[], ['current','profiles'], [], True], # need to determine if pretrain
            # [[], ['current','profiles'], ['first'], False],
            # [[], ['current','profiles'], ['first'], True],
            # [[], ['current','profiles'], ['last'], False],
            # [[], ['current','profiles'], ['last'], True],
            # [[], ['current','profiles'], ['first','last'], False],
            # [[], ['current','profiles'], ['first','last'], True],
    return exp_config


class train_val_test():
    def __init__(self, device, path_prepared, dataset,
                 encoder_selection='all', 
                 cross_attention=[],
                 pretrained_encoder=False,
                 return_attention=False):
        super(train_val_test, self).__init__()
        self.device = device
        self.path_prepared = path_prepared
        self.dataset = dataset
        dataset_name = '_'.join(dataset)
        self.dataset_name = dataset_name
        if encoder_selection == 'all':
            encoder_selection = ['current', 'environment', 'profiles']
        encoder_name = '_'.join(encoder_selection)
        self.encoder_name = encoder_name
        cross_attention_name = '_'.join(cross_attention) if len(cross_attention) > 0 else 'not_crossed'
        self.cross_attention_name = cross_attention_name
        if not return_attention:
            if pretrained_encoder:
                self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/pretrained/{encoder_name}_{cross_attention_name}/'
            else:
                self.path_output = path_prepared + f'PosteriorInference/{dataset_name}/not_pretrained/{encoder_name}_{cross_attention_name}/'
            os.makedirs(self.path_output, exist_ok=True)
        self.encoder_selection = encoder_selection
        self.cross_attention = cross_attention
        self.pretrained_encoder = pretrained_encoder
        self.return_attention = return_attention
        self.define_model()
        self.loss_func = LogNormalNLL()

    def define_model(self,):
        self.model = UnifiedProximity(self.device, self.encoder_selection, self.cross_attention, self.return_attention)
        if self.pretrained_encoder:
            self.model.load_pretrained_encoders(self.path_prepared)

    def create_dataloader(self, batch_size):
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(DataOrganiser('train', self.dataset, self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(DataOrganiser('val', self.dataset, self.encoder_selection, self.path_prepared), batch_size=self.batch_size, shuffle=False)
        
    def send_x_to_device(self, x):
        if isinstance(x, list):
            return [i.to(self.device) for i in x]
        else:
            return x.to(self.device)

    def train_model(self, num_epochs=300, initial_lr=0.001, lr_schedule=True, verbose=0):
        self.initial_lr = initial_lr
        self.verbose = verbose

        # Move model and loss function to device
        self.model = self.model.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        # Training
        num_batches = len(self.train_dataloader)
        loss_log = np.ones((num_epochs, num_batches))*np.inf
        val_loss_log = []

        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr)

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
            progress_bar = tqdm(range(num_epochs), unit='epoch', ascii=True, dynamic_ncols=False)
        else:
            progress_bar = range(num_epochs)

        break_flag = False
        for epoch_n in progress_bar:
            for train_batch_iter, (x, y) in enumerate(self.train_dataloader):
                out = self.model(self.send_x_to_device(x))
                loss = self.loss_func(out, y.to(self.device))
                loss_log[epoch_n, train_batch_iter] = loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            val_loss = self.val_loop()
            if lr_schedule:
                self.scheduler.step(val_loss)
            val_loss_log.append(val_loss)

            # Add information to progress bar with learning rate and loss values
            if self.verbose > 0:
                progress_bar.set_postfix(lr=self.optimizer.param_groups[0]['lr'],
                                            train_loss=loss_log[epoch_n].mean(), val_loss=val_loss, refresh=False)
                if epoch_n % self.verbose < 1:
                    progress_bar.update(self.verbose)

            # Early stopping if validation loss converges
            if epoch_n > 15:
                stop_condition = np.all(abs(np.diff(val_loss_log)[-4:]/val_loss_log[-4:])<1e-3)
                if stop_condition:
                    break_flag = True
            if break_flag:
                print(f'Validation loss converges and training stops early at Epoch {epoch_n}.')
                break

        self.val_loss_log = np.array(val_loss_log)
        if lr_schedule:
            # Save model and loss records
            torch.save(self.model.state_dict(), self.path_output+f'model_final_{epoch_n}epoch.pth')
            loss_log = loss_log[loss_log.mean(axis=1)<np.inf]
            loss_log = pd.DataFrame(loss_log, index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                    columns=[f'iter_{i}' for i in range(1, len(loss_log[0])+1)])
            loss_log.to_csv(self.path_output+'loss_log.csv')
            val_loss_log = pd.DataFrame(val_loss_log[11:], index=[f'epoch_{i}' for i in range(1, len(val_loss_log)-10)], columns=['val_loss'])
            val_loss_log.to_csv(self.path_output+'val_loss_log.csv')

    # Validation loop
    def val_loop(self,):
        self.model.eval()
        val_loss = np.zeros(len(self.val_dataloader))
        with torch.no_grad():
            for val_batch_iter, (x, y) in enumerate(self.val_dataloader):
                out = self.model(self.send_x_to_device(x))
                loss = self.loss_func(out, y.to(self.device)).item()
                val_loss[val_batch_iter] = loss
        self.model.train()
        return val_loss.mean()
    
    def load_model(self, batch_size=None, initial_lr=None):
        if 'path_output' not in self.__dict__:
            if self.pretrained_encoder:
                self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/pretrained/{self.encoder_name}_{self.cross_attention_name}/'
            else:
                self.path_output = self.path_prepared + f'PosteriorInference/{self.dataset_name}/not_pretrained/{self.encoder_name}_{self.cross_attention_name}/'
        if batch_size is not None and initial_lr is not None:
            self.path_save = self.path_output + f'bs={batch_size}-initlr={initial_lr}/'
        else:
            self.path_save = self.path_output
        final_model = glob.glob(self.path_save+'model_final*')[0]
        self.model.load_state_dict(torch.load(final_model, map_location=torch.device(self.device), weights_only=True))        
        self.model = self.model.to(self.device)
        self.loss_func = self.loss_func.to(self.device)
        self.model.eval()

    # def test_model(self, batch_size=None, initial_lr=None):
    #     # Load trained model
    #     self.load_model(batch_size, initial_lr)

    #     # Evaluate the model
    #     self.model.eval()
    #     test_loss = np.zeros(len(self.test_dataloader))
    #     with torch.no_grad():
    #         for test_batch_iter, (x, y) in enumerate(self.test_dataloader):
    #             out = self.model(self.send_x_to_device(x))
    #             loss = self.loss_func(out, y.to(self.device)).item()
    #             test_loss[test_batch_iter] = loss
    #     self.model.train()
    #     return test_loss.mean()
