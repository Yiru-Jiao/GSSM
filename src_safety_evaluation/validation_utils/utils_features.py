'''
This script defines functions to prepare data for safety evaluation and analysis.
'''

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()

manual_seed = 131
path_prepared = './PreparedData/'
path_processed = './ProcessedData/SHRP2/'


def read_data(event_cat, single_file=True, path_processed=path_processed):
    data_ego = pd.read_hdf(path_processed + event_cat + '/Ego_birdseye.h5', key='data')
    data_ego['hx'] = np.cos(data_ego['psi_ekf'])
    data_ego['hy'] = np.sin(data_ego['psi_ekf'])

    data_sur = pd.read_hdf(path_processed + event_cat + '/Surrounding_birdseye.h5', key='data')
    data_sur['hx'] = np.cos(data_sur['psi_ekf'])
    data_sur['hy'] = np.sin(data_sur['psi_ekf'])

    data_ego = data_ego[['time','event_id','x_ekf','y_ekf','v_ekf','omega_ekf','acc_ekf','hx','hy']]
    data_ego = data_ego.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v','omega_ekf':'omega_ego','acc_ekf':'acc_ego'})
    data_sur = data_sur[['time','event_id','target_id','x_ekf','y_ekf','v_ekf','hx','hy']]
    data_sur = data_sur.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v'})

    if single_file:
        data_both = data_ego.merge(data_sur, on=['event_id','time'], how='inner', suffixes=('_ego', '_sur'))
        data_both = data_both.sort_values(['target_id','time']).set_index(['target_id','time'])
        return data_both
    else:
        return data_ego, data_sur


def get_scaler(datasets, path_prepared, feature='profiles'):
    print(f'Getting scaler for {feature}...')
    if feature == 'profiles':
        scaler_data = []
        for dataset in datasets:
            for split in ['train', 'val']:
                scaler_data.append(pd.read_hdf(f'{path_prepared}Segments/{dataset}/profiles_{dataset}_{split}.h5', key='profiles'))
        scaler_data = pd.concat(scaler_data, ignore_index=True)
        scaler_data = scaler_data[['v_ego','v_sur','angle']].values
        scaler = StandardScaler().fit(scaler_data)
    elif feature == 'current':
        variables = ['l_ego','l_sur','delta_v','psi_sur','acc_ego','v_ego2','v_sur2','delta_v2','rho']
        scaler_data = []
        for dataset in datasets:
            for split in ['train', 'val']:
                scaler_data.append(pd.read_hdf(f'{path_prepared}Segments/{dataset}/current_features_{dataset}_{split}.h5', key='features'))
        scaler_data = pd.concat(scaler_data, ignore_index=True)
        scaler_data = scaler_data[variables].values
        scaler = StandardScaler().fit(scaler_data)
    elif feature == 'environment':
        print('No scaler is needed for environment features.')
    return scaler


def segment_data(df, ego_length, target_length):
    df_view_ego = coortrans.transform_coor(df, 'ego')
    df_view_relative = coortrans.transform_coor(df, 'relative')
    indices_end = np.arange(len(df)-1, 20, -1) # use 2-second history for every 0.1 second
    profiles_set = []
    current_features_set = []
    spacing_set = []
    for idx_end in indices_end:
        profiles = df.iloc[idx_end-20:idx_end][['v_ego','omega_ego','v_sur']] # speed and yaw rate of ego, speed of surrounding
        assert profiles.isna().sum().sum()==0 # no missing values

        current_features = np.zeros(9)
        current_features[0] = ego_length
        current_features[1] = target_length
        vx_ego = df.iloc[idx_end]['v_ego']*df.iloc[idx_end]['hx_ego']
        vy_ego = df.iloc[idx_end]['v_ego']*df.iloc[idx_end]['hy_ego']
        vx_sur = df.iloc[idx_end]['v_sur']*df.iloc[idx_end]['hx_sur']
        vy_sur = df.iloc[idx_end]['v_sur']*df.iloc[idx_end]['hy_sur']
        current_features[2] = np.sqrt((vx_ego-vx_sur)**2 + (vy_ego-vy_sur)**2) * np.sign(df.iloc[idx_end]['v_ego']-df.iloc[idx_end]['v_sur'])
        current_features[3] = coortrans.angle(1, 0, df_view_ego.iloc[idx_end]['hx_sur'], df_view_ego.iloc[idx_end]['hy_sur'])
        current_features[4] = df.iloc[idx_end]['acc_ego']
        current_features[5] = df.iloc[idx_end]['v_ego']**2
        current_features[6] = df.iloc[idx_end]['v_sur']**2
        current_features[7] = (vx_ego-vx_sur)**2 + (vy_ego-vy_sur)**2
        current_features[8] = coortrans.angle(1, 0, df_view_relative.iloc[idx_end]['x_sur'], df_view_relative.iloc[idx_end]['y_sur'])
        spacing = np.sqrt(df_view_relative.iloc[idx_end]['x_sur']**2 + df_view_relative.iloc[idx_end]['y_sur']**2)

        profiles_set.append(profiles.values)
        current_features_set.append(current_features)
        spacing_set.append(spacing)
    index_set = df.iloc[indices_end].reset_index()[['event_id','target_id','time']].values
    return np.array(profiles_set), np.array(current_features_set), np.array(spacing_set), index_set


def get_context_representations(df, ego_length, target_length):
    profiles_set, current_features_set, spacing_set, index_set = segment_data(df, ego_length, target_length)
    assert np.isnan(profiles_set).sum()==0
    return profiles_set, current_features_set, spacing_set, index_set

