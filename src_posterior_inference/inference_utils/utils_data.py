'''

'''

import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def get_scaler(dataset_dir, feature):
    print(f'Getting scaler for {feature}...')
    if feature == 'profiles':
        scaler_data = pd.concat([pd.read_hdf(f'{dataset_dir}SafeBaselines/profiles_{split}.h5', key='profiles') for split in ['train', 'val', 'test']], ignore_index=True)
        scaler_data = scaler_data[['v_ego','omega_ego','v_sur']].values
        scaler = StandardScaler()
        scaler.fit(scaler_data)
    elif feature == 'current':
        variables = ['v_ego','v_sur','delta_v','psi_sur','acc_ego','v_ego2','v_sur2','delta_v2','rho']
        scaler_data = pd.concat([pd.read_hdf(f'{dataset_dir}SafeBaselines/current_features_{split}.h5', key='features') for split in ['train', 'val', 'test']], ignore_index=True)
        scaler_data = scaler_data[variables].values
        scaler = StandardScaler()
        scaler.fit(scaler_data)
    return scaler


class DataOrganiser(Dataset):
    def __init__(self, split, encoder_selection, path_prepared):
        super(DataOrganiser, self).__init__()
        self.split = split
        if encoder_selection=='all':
            encoder_selection = ['current', 'environment', 'profiles']
        self.encoder_selection = encoder_selection
        self.path_prepared = path_prepared
        self.current_scaler = get_scaler(path_prepared, 'current')
        if 'profiles' in encoder_selection:
            self.profiles_scaler = get_scaler(path_prepared, 'profiles')
        self.data = self.read_data()
        self.combine_features = self.define_combine_features()

    def __len__(self,):
        return len(self.data[-1])

    def __getitem__(self, idx):
        x, y = self.combine_features(idx)
        return x, y

    def define_combine_features(self,):
        if self.encoder_selection==['current']:
            def combine_features(idx):
                return self.data[0][idx], self.data[-1][idx]
        elif self.encoder_selection==['current','environment'] or self.encoder_selection==['current','profiles']:
            def combine_features(idx):
                return (self.data[0][idx], self.data[1][idx]), self.data[-1][idx]
        elif self.encoder_selection==['current','environment','profiles']:
            def combine_features(idx):
                return (self.data[0][idx], self.data[1][idx], self.data[2][idx]), self.data[-1][idx]
        return combine_features

    def read_data(self,):
        print(f'Reading data for {self.encoder_selection} {self.split}...')
        self.data = []
        X_current = pd.read_hdf(os.path.join(self.path_prepared, 'SafeBaselines/current_features_'+self.split+'.h5'), key='features')
        self.scene_ids = X_current['scene_id'].values
        variables = ['v_ego','v_sur','delta_v','psi_sur','acc_ego','v_ego2','v_sur2','delta_v2','rho']
        self.data.append(torch.from_numpy(self.current_scaler.transform(X_current[variables].values)).float())

        if 'environment' in self.encoder_selection:
            X_environment = pd.read_hdf(os.path.join(self.path_prepared, 'SafeBaselines/environment_features_'+self.split+'.h5'), key='features')
            X_environment = X_environment.sort_values('scene_id').reset_index(drop=True)
            assert np.all(X_current['scene_id'].values==X_environment['scene_id'].values)
            X_environment = X_environment.drop(columns=['scene_id', 'event_id', 'target_id'])
            assert X_environment.shape[1]==27
            self.data.append(torch.from_numpy(X_environment.values).float())
        
        if 'profiles' in self.encoder_selection:
            X_profiles = pd.read_hdf(os.path.join(self.path_prepared, 'SafeBaselines/profiles_'+self.split+'.h5'), key='profiles')
            X_profiles = X_profiles.sort_values(['scene_id', 'time']).reset_index(drop=True)
            assert np.all(X_current['scene_id'].values==X_profiles['scene_id'].drop_duplicates().values)
            X_profiles = X_profiles[['v_ego','omega_ego','v_sur']].values.reshape(-1, 20, 3)
            X_profiles = self.profiles_scaler.transform(X_profiles.reshape(-1, 3)).reshape(X_profiles.shape)
            self.data.append(torch.from_numpy(X_profiles).float())

        if np.any(X_current['s']<=1e-6): # the spacing must be larger than 0
            warnings.warn('There are spacings smaller than or equal to 0.')
            X_current.loc[X_current['s']<=1e-6, 's'] = 1e-6
        self.data.append(torch.from_numpy(X_current[['s']].values).float())

        return self.data
