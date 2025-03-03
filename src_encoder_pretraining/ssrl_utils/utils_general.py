'''
This script includes general utility functions that are used in the training and evaluation scripts.
'''

import os
import numpy as np
import pandas as pd
import torch
import random


def save_checkpoint_callback(save_dir, save_every=0, unit='epoch'):
    assert unit in ('epoch', 'iter')

    if save_every == 0:
        def callback(model, finish=False):
            if finish:
                n = model.epoch_n if unit == 'epoch' else model.iter_n
                model.save(os.path.join(save_dir, f'model_final_{n}{unit}'))
    else:
        def callback(model, finish=False):
            """
            Callback function that saves the model checkpoint. 
            Save the model at the initial iteration/epoch and the parameters thereafter.
            """
            if finish:
                model.save(os.path.join(save_dir, f'model_final_{n}{unit}'))
            else:
                n = model.epoch_n if unit == 'epoch' else model.iter_n
                if n > 0 and n % save_every == 0:
                    model.save(os.path.join(save_dir, f'model_{n}{unit}'))

    return callback


def fix_seed(seed, deterministic=False):
    random.seed(seed)
    seed += 1
    np.random.seed(seed)
    seed += 1
    torch.manual_seed(seed)
    seed += 1
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def init_dl_program(gpu_num=0, max_threads=None, use_tf32=False):

    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        else:
            try:
                import mkl
                mkl.set_num_threads(max_threads)
            except:
                pass
        
    if isinstance(gpu_num, (str, int)):
        if gpu_num == '0':
            device_name = ['cpu']
        elif ',' in gpu_num:
            device_name = ['cuda:'+idx for idx in gpu_num.split(',')]
            # Reduce VRAM usage by reducing fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        else:
            device_name = [f'cuda:{idx}' for idx in range(int(gpu_num))]
            # Reduce VRAM usage by reducing fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    devices = []
    for device in reversed(device_name):
        torch_device = torch.device(device)
        devices.append(torch_device)
        if torch_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(torch_device)
    devices.reverse()

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = use_tf32
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]


def get_tuned_params(args, tuned_params, phase, param_names):
    for param_name in param_names:
        if param_name not in tuned_params.columns:
            raise ValueError(f'Invalid parameter name {param_name}.')
        else:
            param_value = tuned_params.loc[phase, param_name]
            if param_name == 'temporal_hierarchy':
                if param_value != 'linear' and param_value != 'exponential':
                    param_value = None
            if param_name == 'batch_size':
                param_value = int(param_value)
            elif param_name in ['weight_lr', 'bandwidth', 'tau_inst', 'tau_temp']:
                param_value = float(param_value)
            setattr(args, param_name, param_value)
    return args


def load_tuned_hyperparameters(args, tuned_params, model_type=None):
    # Set default parameters
    args.tau_inst = 0
    args.tau_temp = 0
    args.temporal_hierarchy = None
    args.regularizer = None
    args.bandwidth = 1.
    args.batch_size = 8
    args.weight_lr = 0.01

    if model_type is None:
        if args.tau_inst == 0 and args.tau_temp == 0:
            if args.regularizer is None:
                model_type = 'ts2vec'
            elif args.regularizer == 'topology':
                model_type = 'topo-ts2vec'
            elif args.regularizer == 'geometry':
                model_type = 'ggeo-ts2vec'
        else:
            if args.regularizer is None:
                model_type = 'softclt'
            elif args.regularizer == 'topology':
                model_type = 'topo-softclt'
            elif args.regularizer == 'geometry':
                model_type = 'ggeo-softclt'
    if model_type == 'ts2vec':
        param_names = ['batch_size']
        args = get_tuned_params(args, tuned_params, 'TS2Vec_Phase1', param_names)
        args.regularizer = None
    elif model_type == 'topo-ts2vec':
        param_names = ['batch_size', 'weight_lr']
        args = get_tuned_params(args, tuned_params, 'TopoTS2Vec_Phase1', param_names)
        args.regularizer = 'topology'
    elif model_type == 'ggeo-ts2vec':
        param_names = ['batch_size', 'weight_lr', 'bandwidth']
        args = get_tuned_params(args, tuned_params, 'GGeoTS2Vec_Phase1', param_names)
        args.regularizer = 'geometry'
    elif model_type == 'softclt':
        param_names = ['tau_inst', 'tau_temp', 'temporal_hierarchy', 'batch_size']
        args = get_tuned_params(args, tuned_params, 'SoftCLT_Phase2', param_names)
        args.regularizer = None
    elif model_type == 'topo-softclt':
        param_names = ['tau_inst', 'tau_temp', 'temporal_hierarchy', 'batch_size', 'weight_lr']
        args = get_tuned_params(args, tuned_params, 'TopoSoftCLT_Phase1', param_names)
        args.regularizer = 'topology'
    elif model_type == 'ggeo-softclt':
        param_names = ['tau_inst', 'tau_temp', 'temporal_hierarchy', 'batch_size', 'weight_lr', 'bandwidth']
        args = get_tuned_params(args, tuned_params, 'GGeoSoftCLT_Phase1', param_names)
        args.regularizer = 'geometry'
    return args


def configure_model(args, input_dims, device):
    # Define loss configuration
    loss_config = dict(
        tau_inst=args.tau_inst,
        tau_temp=args.tau_temp,
        temporal_hierarchy=args.temporal_hierarchy,
        )

    # Define regularizer configuration
    regularizer_config = dict(
        reserve=args.regularizer,
        bandwidth=args.bandwidth,
    )

    # Define model configuration
    model_config = dict(
        input_dims=input_dims,
        output_dims=args.repr_dims,
        dist_metric=args.dist_metric,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_lr=args.weight_lr,
        loss_config=loss_config,
        regularizer_config=regularizer_config,
        )
    
    return model_config


def save_loss_log(loss_log, save_dir, regularizer=None):
    if loss_log.shape[-1] == 1:
        loss_log = pd.DataFrame(loss_log, index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)], columns=['train_loss'])
    elif loss_log.shape[-1] == 3:
        loss_log = pd.DataFrame(loss_log, index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)], 
                                columns=['train_loss', 'val_loss', 'regularizer'])
    elif loss_log.shape[-1] == 7:
        if regularizer == 'topology':
            loss_log = pd.DataFrame(loss_log, index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                    columns=['train_loss', 'loss_scl', 'log_var_scl', 'loss_topo', 'log_var_topo', 'val_loss', 'regularizer'])
        elif regularizer == 'geometry':
            loss_log = pd.DataFrame(loss_log, index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                    columns=['train_loss', 'loss_scl', 'log_var_scl', 'loss_ggeo', 'log_var_ggeo', 'val_loss', 'regularizer'])
    loss_log.to_csv(f'{save_dir}/loss_log.csv')
