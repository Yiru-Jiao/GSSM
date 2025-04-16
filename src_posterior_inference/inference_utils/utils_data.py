'''
This script contains the functions to organise data for posterior inference.
'''

import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import Dataset


class DataOrganiser(Dataset):
    def __init__(self, split, dataset, encoder_selection, path_prepared, mixrate=2, random_seed=131):
        super(DataOrganiser, self).__init__()
        self.split = split
        self.dataset = dataset
        if encoder_selection=='all':
            encoder_selection = ['current+acc', 'environment', 'profiles']
        self.encoder_selection = encoder_selection
        self.path_prepared = path_prepared
        self.mixrate = mixrate
        self.random_seed = random_seed
        self.data = self.read_data()
        self.combine_features = self.define_combine_features()

    def __len__(self,):
        return len(self.data[-1])

    def __getitem__(self, idx):
        x, y = self.combine_features(idx)
        return x, y

    def define_combine_features(self,):
        if len(self.encoder_selection)==1:
            def combine_features(idx):
                return self.data[0][idx], self.data[-1][idx]
        elif len(self.encoder_selection)==2:
            def combine_features(idx):
                return tuple([self.data[0][idx], self.data[1][idx]]), self.data[-1][idx]
        elif len(self.encoder_selection)==3:
            def combine_features(idx):
                return tuple([self.data[0][idx], self.data[1][idx], self.data[2][idx]]), self.data[-1][idx]
        return combine_features

    def read_data(self,):
        print(f'Reading data for {self.dataset} {self.encoder_selection} {self.split}...')
        mixrate_dict = {'SafeBaseline': 2, 'ArgoverseHV': 2, 'highD': 2}
        if len(self.dataset)==2 and self.mixrate<1:
            mixrate_dict[self.dataset[1]] = self.mixrate
        elif len(self.dataset)==3:
            mixrate_dict['ArgoverseHV'] = 0.4
            mixrate_dict['highD'] = 0.4
        self.data = []
        X_current = []
        scene_id = 0
        for dataset in self.dataset:
            x_current = pd.read_hdf(f'{self.path_prepared}Segments/{dataset}/current_features_{dataset}_{self.split}.h5', key='features')
            x_current['scene_id'] = x_current['scene_id'] + scene_id
            scene_id = x_current['scene_id'].max() + 1
            if mixrate_dict[dataset] >= 1:
                X_current.append(x_current)
            else:
                mixrate = mixrate_dict[dataset]
                scene_ids = x_current['scene_id'].unique()
                scene_ids = np.random.RandomState(self.random_seed).choice(scene_ids, int(len(scene_ids)*mixrate), replace=False)
                X_current.append(x_current[x_current['scene_id'].isin(scene_ids)])

        X_current = pd.concat(X_current, ignore_index=True)
        X_current = X_current.sort_values('scene_id').reset_index(drop=True)
        self.scene_ids = X_current['scene_id'].values
        if 'acc' in self.encoder_selection[0]:
            variables = ['l_ego','l_sur','combined_width',
                         'vy_ego','vx_sur','vy_sur','v_ego2','v_sur2','delta_v2','delta_v',
                         'psi_sur','acc_ego','rho']
        else:
            variables = ['l_ego','l_sur','combined_width',
                         'vy_ego','vx_sur','vy_sur','v_ego2','v_sur2','delta_v2','delta_v',
                         'psi_sur','rho']
        self.data.append(torch.from_numpy(X_current[variables].values).float())

        if 'environment' in self.encoder_selection:
            X_environment = []
            scene_id = 0
            for dataset in self.dataset:
                x_environment = pd.read_hdf(f'{self.path_prepared}Segments/{dataset}/environment_features_{dataset}_{self.split}.h5', key='features')
                x_environment['scene_id'] = x_environment['scene_id'] + scene_id
                scene_id = x_environment['scene_id'].max() + 1
                X_environment.append(x_environment)
            X_environment = pd.concat(X_environment, ignore_index=True)
            X_environment = X_environment.sort_values('scene_id').reset_index(drop=True)
            assert np.all(X_current['scene_id'].values==X_environment['scene_id'].values)
            X_environment = X_environment.drop(columns=['scene_id', 'event_id', 'target_id'])
            assert X_environment.shape[1]==27
            self.data.append(torch.from_numpy(X_environment.values).float())
        
        if 'profiles' in self.encoder_selection:
            X_profiles = []
            scene_id = 0
            for dataset in self.dataset:
                x_profiles = pd.read_hdf(f'{self.path_prepared}Segments/{dataset}/profiles_{dataset}_{self.split}.h5', key='profiles')
                x_profiles['scene_id'] = x_profiles['scene_id'] + scene_id
                scene_id = x_profiles['scene_id'].max() + 1
                X_profiles.append(x_profiles)
            X_profiles = pd.concat(X_profiles)
            X_profiles = X_profiles.sort_values(['scene_id', 'time']).reset_index(drop=True)
            assert np.all(X_current['scene_id'].values==X_profiles['scene_id'].drop_duplicates().values)
            X_profiles = X_profiles[['yaw_ego','v_ego','vx_sur','vy_sur']].values.reshape(-1, 25, 4)
            self.data.append(torch.from_numpy(X_profiles).float())

        if np.any(X_current['s']<=1e-6): # the spacing must be larger than 0
            warnings.warn('There are spacings smaller than or equal to 0.')
            X_current.loc[X_current['s']<=1e-6, 's'] = 1e-6
        self.data.append(torch.from_numpy(X_current['s'].values).float())

        return tuple(self.data)
