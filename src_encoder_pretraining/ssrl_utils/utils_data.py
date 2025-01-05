'''
This file contains functions to load and preprocess datasets for training and evaluation.
'''

import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.ssrl_utils.utils_distance_matrix import *


def set_nan_to_zero(a):
    a[np.isnan(a)] = 0
    return a


def normalize_TS(TS):
    TS = set_nan_to_zero(TS)
    if TS.ndim == 2: # univariate
        TS_max = TS.max(axis = 1).reshape(-1,1)
        TS_min = TS.min(axis = 1).reshape(-1,1)
        TS = (TS - TS_min)/(TS_max - TS_min + (1e-6))
    elif TS.ndim == 3: # multivariate
        N, D, L = TS.shape
        TS_max = TS.max(axis=2).reshape(N,D,1) 
        TS_min = TS.min(axis=2).reshape(N,D,1)
        TS = (TS - TS_min) / (TS_max - TS_min + (1e-6))
    return TS


def compute_sim_mat(data, dist_metric='DTW', min_=0, max_=1):
    if data.ndim == 3: # (n_instance, n_timestamps, n_features)
        if data.shape[2] == 1:
            multivariate = False
            data = data.reshape(data.shape[0], -1)
        else:
            multivariate = True
        norm_TS = normalize_TS(data)
        sim_mat = save_sim_mat(norm_TS, multivariate, dist_metric, min_=min_, max_=max_)
    elif data.ndim == 4: # multivariate with multiple channels (n_instance, n_timestamps, n_nodes/agents, n_features)
        if data.shape[3] == 1:
            data = data.reshape(data.shape[0], data.shape[1], -1)
            sim_mat = compute_sim_mat(data, dist_metric, min_, max_)
        else:
            sim_mat = np.zeros((data.shape[0], data.shape[0]))
            for channel_idx in range(data.shape[3]):
                sim_mat += compute_sim_mat(data[..., channel_idx], dist_metric, min_, max_)
            sim_mat /= data.shape[3]
    return sim_mat


def load_data(dataset_dir='./PreparedData/', feature='profiles'):
    if feature == 'profiles':
        train_data = pd.read_hdf(f'{dataset_dir}SafeBaselines/profiles_train_downsampled.h5', key='profiles')
        train_X = train_data[['v_ego','omega_ego','v_sur']].values.reshape(-1, 20, 3)
        val_data = pd.read_hdf(f'{dataset_dir}SafeBaselines/profiles_val_downsampled.h5', key='profiles')
        val_X = val_data[['v_ego','omega_ego','v_sur']].values.reshape(-1, 20, 3)

        scaler_data = pd.concat([pd.read_hdf(f'{dataset_dir}SafeBaselines/profiles_{split}.h5', key='profiles') for split in ['train', 'val', 'test']], ignore_index=True)
        scaler_data = scaler_data[['v_ego','omega_ego','v_sur']].values
        scaler = StandardScaler()
        scaler.fit(scaler_data)
        train_X = scaler.transform(train_X.reshape(-1, 3)).reshape(train_X.shape)
        val_X = scaler.transform(val_X.reshape(-1, 3)).reshape(val_X.shape)

        assert train_X.ndim == 3 # (n_instance, n_timestamps, n_features)
        assert val_X.ndim == 3
    elif feature == 'current':
        variables = ['v_ego','v_sur','delta_v','psi_sur','acc_ego','v_ego2','v_sur2','delta_v2','rho']
        train_data = pd.read_hdf(f'{dataset_dir}SafeBaselines/current_features_train_downsampled.h5', key='features')
        train_X = train_data[variables].values
        val_data = pd.read_hdf(f'{dataset_dir}SafeBaselines/current_features_val_downsampled.h5', key='features')
        val_X = val_data[variables].values

        scaler_data = pd.concat([pd.read_hdf(f'{dataset_dir}SafeBaselines/current_features_{split}.h5', key='features') for split in ['train', 'val', 'test']], ignore_index=True)
        scaler_data = scaler_data[variables].values
        scaler = StandardScaler()
        scaler.fit(scaler_data)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
    elif feature == 'environment':
        train_data = pd.read_hdf(f'{dataset_dir}SafeBaselines/environment_features_train_downsampled.h5', key='features')
        train_X = train_data.values
        val_data = pd.read_hdf(f'{dataset_dir}SafeBaselines/environment_features_val_downsampled.h5', key='features')
        val_X = val_data.values
    
    return train_X, val_X


def assign_soft_labels(sim_mat, tau_inst):
    if tau_inst <= 0:
        soft_labels = None
    else:
        if sim_mat is None:
            soft_labels = 'compute'
        else:
            tau_inst = float(tau_inst)
            alpha = 0.5
            soft_labels = (2*alpha) / (1 + np.exp(tau_inst*abs(1 - sim_mat))) + (1-alpha)*np.eye(sim_mat.shape[0])
    return soft_labels


class custom_dataset(Dataset): 
    def __init__(self, X):
        self.X = X

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx): 
        X = self.X[idx]
        return X, idx

