'''
This script defines models for structure-preserving time series contrastive learning.
The backbone is adapted from TS2Vec https://github.com/zhihanyue/ts2vec and SoftCLT https://github.com/seunghan96/softclt
We don't use the original TSEncoder because it's very slow for large-scale data
'''

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.modules import encoder, losses
import src_encoder_pretraining.ssrl_utils.utils_data as datautils


class spclt():
    def __init__(self,
        input_dims=4, output_dims=256, hidden_dims=128, depth=4, mask_mode=None,
        dist_metric='DTW', device='cpu', lr=0.001, weight_lr=0.05, batch_size=8,
        after_iter_callback=None, after_epoch_callback=None,
        regularizer_config={'reserve': None, 'topology': 0.0, 'geometry': 0.0},
        loss_config=None,
        ):
        """
        Initialize the spclt model.
        """
        super(spclt, self).__init__()
        self.dist_metric = dist_metric
        self.device = device
        self.lr = lr
        self.weight_lr = weight_lr
        self.batch_size = batch_size
        self.loss_config = loss_config
        self.regularizer_config = regularizer_config
                
        # define encoder
        # self._net = encoder.TSEncoder(input_dims=input_dims,
        #                               output_dims=output_dims,
        #                               hidden_dims=hidden_dims,
        #                               depth=depth,
        #                               mask_mode=mask_mode).to(self.device)
        self._net = encoder.LSTMEncoder(input_dims=input_dims,
                                        hidden_dims=20*output_dims,
                                        num_layers=2,
                                        single_output=True).to(self.device)

        self.net = torch.optim.swa_utils.AveragedModel(self._net).to(self.device)
        self.net.update_parameters(self._net)

        # define learner of log variances used for weighing losses
        if self.regularizer_config['reserve'] in ['topology', 'geometry']:
            self.loss_log_vars = torch.nn.Parameter(torch.zeros(2, device=self.device))
        
        # define callback functions
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

    # define eval() and train() functions
    def eval(self,):
        if self.regularizer_config['reserve'] is None:
            self._net.eval()
            self.net.eval()
        else:
            self._net.eval()
            self.net.eval()
            self.loss_log_vars.requires_grad = False

    def train(self,):
        if self.regularizer_config['reserve'] is None:
            self._net.train()
            self.net.train()
        else:
            self._net.train()
            self.net.train()
            self.loss_log_vars.requires_grad = True


    def fit(self, name_data, train_data, soft_assignments=None, n_epochs=None, n_iters=None, scheduler='constant', verbose=0):
        """
        Fit the model to the training data.
        """
        if isinstance(soft_assignments, np.ndarray):
            assert soft_assignments.shape[0] == soft_assignments.shape[1]
            assert train_data.shape[0] == soft_assignments.shape[0]

        self.train()

        # Set default number for n_iters, this is intended for underfitting the model
        if n_iters is None and n_epochs is None:
            num_samples = train_data.shape[0]
            n_iters = num_samples * 32 / self.batch_size
            sample_bounds = [100, 500, 1000, 10000, 100000, 1000000]
            coefs = [2, 1, 1/2, 1/4, 1/8, 1/16]
            for bound, coef in zip(sample_bounds, coefs):
                if num_samples < bound:
                    n_iters = int(n_iters * coef)
                    break
            if num_samples >= sample_bounds[-1]:
                n_iters = int(n_iters / 64)
            print(f'Number of iterations is set to {n_iters}.')

        # define a progress bar
        if n_epochs is not None:
            if verbose:
                progress_bar = tqdm(range(n_epochs), desc=f'Train {name_data} epoch', ascii=True, dynamic_ncols=False)
            else:
                progress_bar = range(n_epochs)
        elif n_iters is not None:
            if verbose:
                progress_bar = tqdm(range(n_iters), desc=f'Train {name_data} iter', ascii=True, dynamic_ncols=False)
            else:
                progress_bar = range(n_iters)
        else:
            ValueError('At least one between n_epochs and n_iters should be specified')
        
        # define optimizer
        self.optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        if self.regularizer_config['reserve'] is not None:
            self.optimizer_weight = torch.optim.AdamW([self.loss_log_vars], lr=self.weight_lr)
            def optimizer_zero_grad():
                self.optimizer.zero_grad()
                self.optimizer_weight.zero_grad()
            def optimizer_step(scaler):
                scaler.step(self.optimizer)
                scaler.step(self.optimizer_weight)
        else:
            def optimizer_zero_grad():
                self.optimizer.zero_grad()
            def optimizer_step(scaler):
                scaler.step(self.optimizer)

        # exclude instances with all missing values
        isnanmat = np.isnan(train_data)
        while isnanmat.ndim > 1:
            isnanmat = isnanmat.all(axis=-1)
        reserved_idx = ~isnanmat

        # define training and validation data
        if scheduler == 'constant':
            train_data = train_data[reserved_idx]            
            if soft_assignments is None:
                train_soft_assignments = None
            elif isinstance(soft_assignments, str):
                train_soft_assignments = 'compute'
            else:
                train_soft_assignments = soft_assignments[reserved_idx][:,reserved_idx].copy()
            del soft_assignments, reserved_idx
        elif scheduler == 'reduced':
            train_val_data = train_data[reserved_idx]
            # randomly split the training data into training and validation sets
            val_indices = np.random.choice(len(train_val_data), int(len(train_val_data)*0.1), replace=False)
            train_indices = np.setdiff1d(np.arange(len(train_val_data)), val_indices)
            train_data, val_data = train_val_data[train_indices].copy(), train_val_data[val_indices].copy()
            if soft_assignments is None:
                train_soft_assignments, val_soft_assignments = None, None
            elif isinstance(soft_assignments, str):
                train_soft_assignments, val_soft_assignments = 'compute', 'compute'
            else:
                train_val_soft_assignments = soft_assignments[reserved_idx][:,reserved_idx]
                train_soft_assignments = train_val_soft_assignments[train_indices][:,train_indices].copy()
                val_soft_assignments = train_val_soft_assignments[val_indices][:,val_indices].copy()
                del train_val_soft_assignments
            del soft_assignments, reserved_idx, train_val_data, train_indices, val_indices
            val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float())
            val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)), shuffle=False, drop_last=True)
            del val_dataset
            
            if n_epochs is not None:
                val_loss_log = np.zeros((n_epochs, 2)) * np.nan
            else:
                raise ValueError('n_epochs should be specified when using reduced scheduler.')
            # define scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.6, patience=4, cooldown=2,
                threshold=1e-3, threshold_mode='rel', min_lr=self.lr*0.6**15
                )
            
            if self.regularizer_config['reserve'] is not None:
                self.scheduler_weight = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_weight, mode='min', factor=0.6, patience=4, cooldown=2,
                    threshold=1e-3, threshold_mode='rel', min_lr=self.weight_lr*0.6**15
                    )
                def scheduler_step(val_loss, regularizer_loss):
                    self.scheduler.step(val_loss)
                    self.scheduler_weight.step(regularizer_loss)
            else:
                def scheduler_step(val_loss, regularizer_loss):
                    self.scheduler.step(val_loss)
        else:
            ValueError("Undefined scheduler: should be either 'constant' or 'reduced'.")

        # create training dataset, dataloader, and loss log
        train_dataset = datautils.custom_dataset(torch.from_numpy(train_data).float())
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        del train_dataset
        train_iters = int(len(train_loader)*0.5) # use 50% of the total iterations per epoch
        if n_epochs is not None:
            if self.regularizer_config['reserve'] is None:
                loss_log = np.zeros((n_epochs, 1)) * np.nan
            elif self.regularizer_config['reserve'] in ['topology', 'geometry']:
                loss_log = np.zeros((n_epochs, 5)) * np.nan
        else:
            loss_log = None

        # define loss function
        self.loss_func = losses.combined_loss

        # training loop
        self.epoch_n = 0
        self.iter_n = 0
        if self.loss_config is None:
            self.loss_config = {'temporal_unit': 0, 'tau_inst': 0}
        else:
            self.loss_config['temporal_unit'] = 0  ## The minimum unit to perform temporal contrast. 
                                                   ## When training on a very long sequence, increasing this helps to reduce the cost of time and memory.
        continue_training = True
        scaler = torch.amp.GradScaler()  # Initialize gradient scaler
        while continue_training:
            train_loss = torch.tensor(0., device=self.device, requires_grad=False)
            if loss_log is not None and self.regularizer_config['reserve'] is not None:
                train_loss_comp = torch.zeros(4, device=self.device, requires_grad=False)
            for train_batch_iter, (x, idx) in enumerate(train_loader):
                if n_epochs is not None and train_batch_iter >= train_iters:
                    break # use 50% of the total iterations per epoch, after 10 epochs 99.90% of the data is used

                if train_soft_assignments is None:
                    soft_labels = None
                elif isinstance(train_soft_assignments, str):
                    batch_sim_mat = datautils.compute_sim_mat(x.numpy(), self.dist_metric)
                    soft_labels = datautils.assign_soft_labels(batch_sim_mat, self.loss_config['tau_inst'])
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                    del batch_sim_mat
                else:
                    soft_labels = train_soft_assignments[idx][:,idx] # (B, B)
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                train_loss_config = self.loss_config.copy()
                train_loss_config['soft_labels'] = soft_labels

                optimizer_zero_grad()
                with torch.amp.autocast(device_type="cuda"):  # Enables Mixed Precision
                    loss, loss_comp = self.loss_func(self, x.to(self.device),
                                                     train_loss_config, 
                                                     self.regularizer_config)

                scaler.scale(loss).backward()
                optimizer_step(scaler)
                scaler.update()
                del train_loss_config # clear memory

                train_loss += loss
                if loss_log is not None and self.regularizer_config['reserve'] is not None:
                    for i in range(4):
                        train_loss_comp[i] += loss_comp[i]
                
                self.net.update_parameters(self._net)

                # save model if callback every several iterations
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self)

                # update progress bar if n_iters is specified
                if n_iters is not None and verbose:
                    if verbose > 6:
                        progress_bar.set_postfix(loss=loss.item(), refresh=False)
                        progress_bar.update(1)
                    else:
                        step = n_iters // (1+verbose*4)
                        if (self.iter_n+1) % step == 0:
                            progress_bar.set_postfix(loss=loss.item(), refresh=False)
                            progress_bar.update(step)

                self.iter_n += 1
                if n_iters is not None and self.iter_n >= n_iters:
                    continue_training = False
                    break

            if n_epochs is not None:
                train_loss = train_loss.item() / (train_batch_iter+1)
                if loss_log is not None:
                    if self.regularizer_config['reserve'] is None:
                        loss_log[self.epoch_n] = [train_loss]
                    elif self.regularizer_config['reserve'] in ['topology', 'geometry']:
                        train_loss_comp = train_loss_comp.detach().cpu().numpy() / (train_batch_iter+1)
                        loss_log[self.epoch_n] = np.concatenate(([train_loss], train_loss_comp))

            # if the scheduler is set to 'reduced', evaluate validation loss and update learning rate
            if scheduler == 'reduced':
                self.eval()
                with torch.no_grad():
                    val_loss = torch.tensor(0., device=self.device, requires_grad=False)
                    for val_batch_iter, (x, idx) in enumerate(val_loader):
                        if val_soft_assignments is None:
                            soft_labels = None
                        elif isinstance(val_soft_assignments, str):
                            batch_sim_mat = datautils.compute_sim_mat(x.numpy(), self.dist_metric)
                            soft_labels = datautils.assign_soft_labels(batch_sim_mat, self.loss_config['tau_inst'])
                            soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                            del batch_sim_mat
                        else:
                            soft_labels = val_soft_assignments[idx][:,idx]
                            soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                        val_loss_config = self.loss_config.copy()
                        val_loss_config['soft_labels'] = soft_labels

                        with torch.amp.autocast(device_type="cuda"):  # Enables Mixed Precision
                            loss, _ = self.loss_func(self, x.to(self.device),
                                                     val_loss_config,
                                                     self.regularizer_config)
                        val_loss += loss
                        del val_loss_config
                    val_loss = val_loss.item() / (val_batch_iter+1)
                if self.regularizer_config['reserve'] is None:
                    val_loss_log[self.epoch_n, 0] = val_loss
                elif self.regularizer_config['reserve'] in ['topology', 'geometry']:
                    val_loss_log[self.epoch_n, 1] = 0.5*self.loss_log_vars.sum().item()
                    val_loss_log[self.epoch_n, 0] = val_loss - val_loss_log[self.epoch_n, 1]

                # update learning rate after cold start of 10 epochs
                if self.epoch_n >= 10:
                    scheduler_step(val_loss_log[self.epoch_n, 0], val_loss_log[self.epoch_n, 1])
                self.train()

            # save model if callback every several epochs
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self)

            # update progress bar if n_epochs is specified
            if n_epochs is not None and verbose:
                if verbose > 6:
                    if scheduler == 'reduced':
                        progress_bar.set_postfix(loss=train_loss, 
                                                 val_loss=val_loss, 
                                                 lr=self.optimizer.param_groups[0]['lr'], refresh=False)
                    else:
                        progress_bar.set_postfix(loss=train_loss, refresh=False)
                    progress_bar.update(1)
                else:
                    step = n_epochs // (1+verbose*4)
                    if (self.epoch_n+1) % step == 0:
                        if scheduler == 'reduced':
                            progress_bar.set_postfix(loss=train_loss, 
                                                     val_loss=val_loss, 
                                                     lr=self.optimizer.param_groups[0]['lr'], refresh=False)
                        else:
                            progress_bar.set_postfix(loss=train_loss, refresh=False)
                        progress_bar.update(step)

            self.epoch_n += 1
            if n_epochs is not None and self.epoch_n >= n_epochs:
                continue_training = False
                break
        
        progress_bar.close()
        if self.after_iter_callback is not None:
            self.after_iter_callback(self, finish=True)
        if self.after_epoch_callback is not None:
            self.after_epoch_callback(self, finish=True)

        if loss_log is None:
            return None
        else:
            if scheduler == 'reduced':
                return np.concatenate((loss_log[:self.epoch_n], val_loss_log[:self.epoch_n]), axis=1)
            else:
                return loss_log[:self.epoch_n]
    

    def compute_loss(self, val_data, soft_assignments, loss_config=None):
        """
        Computes the loss for the given validation data and soft assignments.
        """
        assert self._net is not None, 'please train or load a model first'
        if isinstance(soft_assignments, np.ndarray):
            assert soft_assignments.shape[0] == soft_assignments.shape[1]
            assert val_data.shape[0] == soft_assignments.shape[0] if soft_assignments is not None else True

        # create test dataset and dataloader
        val_dataset = datautils.custom_dataset(torch.from_numpy(val_data).float())
        val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)), shuffle=False, drop_last=True)
        del val_dataset

        # define loss function
        self.loss_func = losses.combined_loss
        
        if self.loss_config is None:
            self.loss_config = {'temporal_unit': 0, 'tau_inst': 0}
        else:
            self.loss_config['temporal_unit'] = 0  ## The minimum unit to perform temporal contrast. 
                                                   ## When training on a very long sequence, increasing this helps to reduce the cost of time and memory.
        if loss_config is None:
            loss_config = self.loss_config
        else:
            loss_config['temporal_unit'] = 0
 
        org_training = self._net.training
        self.eval()
        with torch.no_grad():
            val_loss = torch.tensor(0., device=self.device, requires_grad=False)
            for val_batch_iter, (x, idx) in enumerate(val_loader):
                if soft_assignments is None:
                    soft_labels = None
                elif isinstance(soft_assignments, str):
                    batch_sim_mat = datautils.compute_sim_mat(x.numpy(), self.dist_metric)
                    soft_labels = datautils.assign_soft_labels(batch_sim_mat, loss_config['tau_inst'])
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                    del batch_sim_mat
                else:
                    soft_labels = soft_assignments[idx][:,idx]
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                val_loss_config = loss_config.copy()
                val_loss_config['soft_labels'] = soft_labels

                val_loss, val_loss_comp = self.loss_func(self, x.to(self.device),
                                                         val_loss_config, 
                                                         self.regularizer_config)
                val_loss += val_loss_comp[0]
                del val_loss_config
            val_loss = val_loss.item() / (val_batch_iter+1)
        if org_training:
            self.train()
        return val_loss

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        """
        Evaluate the network output with optional pooling and slicing.
        """
        out = self.net(x.to(self.device), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out


    def torch_pad_nan(arr, left=0, right=0, dim=0):
        if left > 0:
            padshape = list(arr.shape)
            padshape[dim] = left
            arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
        if right > 0:
            padshape = list(arr.shape)
            padshape[dim] = right
            arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
        return arr    
    
    
    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        """
        Encodes the input data using the trained neural network model.
        """
        assert self.net is not None, 'please train or load a net first'
        org_training = self.net.training
        self.net.eval()
        
        assert data.ndim == 3
        n_samples, ts_l, _ = data.shape

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(data, torch.Tensor):
            dataset = datautils.custom_dataset(data)
        else:
            dataset = datautils.custom_dataset(torch.from_numpy(data).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)        

        with torch.no_grad():
            output = []
            for x, _ in dataloader:
                x = x.to(self.device)
                # if sliding_length is not None:
                #     reprs = []
                #     if n_samples < batch_size:
                #         calc_buffer = []
                #         calc_buffer_l = 0
                #     for i in range(0, ts_l, sliding_length):
                #         l = i - sliding_padding
                #         r = i + sliding_length + (sliding_padding if not causal else 0)
                #         x_sliding = self.torch_pad_nan(
                #             x[:, max(l, 0) : min(r, ts_l)],
                #             left=-l if l<0 else 0,
                #             right=r-ts_l if r>ts_l else 0,
                #             dim=1
                #         )
                #         if n_samples < batch_size:
                #             if calc_buffer_l + n_samples > batch_size:
                #                 out = self._eval_with_pooling(
                #                     torch.cat(calc_buffer, dim=0),
                #                     mask,
                #                     slicing=slice(sliding_padding, sliding_padding+sliding_length),
                #                     encoding_window=encoding_window
                #                 )
                #                 reprs += torch.split(out, n_samples)
                #                 calc_buffer = []
                #                 calc_buffer_l = 0
                #             calc_buffer.append(x_sliding)
                #             calc_buffer_l += n_samples
                #         else:
                #             out = self._eval_with_pooling(
                #                 x_sliding,
                #                 mask,
                #                 slicing=slice(sliding_padding, sliding_padding+sliding_length),
                #                 encoding_window=encoding_window
                #             )
                #             reprs.append(out)

                #     if n_samples < batch_size:
                #         if calc_buffer_l > 0:
                #             out = self._eval_with_pooling(
                #                 torch.cat(calc_buffer, dim=0),
                #                 mask,
                #                 slicing=slice(sliding_padding, sliding_padding+sliding_length),
                #                 encoding_window=encoding_window
                #             )
                #             reprs += torch.split(out, n_samples)
                #             calc_buffer = []
                #             calc_buffer_l = 0
                    
                #     out = torch.cat(reprs, dim=1)
                #     if encoding_window == 'full_series':
                #         out = F.max_pool1d(
                #             out.transpose(1, 2).contiguous(),
                #             kernel_size = out.size(1),
                #         ).squeeze(1)
                # else:
                #     out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                #     if encoding_window == 'full_series':
                #         out = out.squeeze(1)

                out = self.net(x) # (batch_size, seq_length, output_dim)
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output


    def save(self, fn):
        """
        Save the model's state dictionary and loss log variables to files.

        Parameters:
        fn (str): The base filename to which the model's state dictionary and loss log variables will be saved.
                  The state dictionary will be saved with the suffix '_net.pth'.
                  If the 'reserve' key in the regularizer configuration is not None, the log variance will be saved
                  with the suffix '_loss_log_vars.npy'.

        Returns:
        None
        """
        torch.save(self.net.state_dict(), fn+'_net.pth')
        if self.regularizer_config['reserve'] is not None:
            state_loss_log_vars = self.loss_log_vars.detach().cpu().numpy()
            np.save(fn+'_loss_log_vars.npy', state_loss_log_vars)


    def load(self, fn):
        """
        Load the model state and associated parameters from the specified file.
        """
        state_dict = torch.load(fn+'_net.pth', map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self._net = self.net
        self.net.eval()
        self._net.eval()
        if self.regularizer_config['reserve'] is not None:
            state_loss_log_vars = np.load(fn+'_loss_log_vars.npy')
            state_loss_log_vars = torch.from_numpy(state_loss_log_vars).to(self.device)
            self.loss_log_vars = torch.nn.Parameter(state_loss_log_vars, requires_grad=False)
