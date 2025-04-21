'''
This script includes general utility functions that are used in the training and evaluation scripts.
'''

import os
import numpy as np
import pandas as pd
import torch
import random


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


