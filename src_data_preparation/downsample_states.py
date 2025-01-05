'''
'''

import os
import sys
from tqdm import tqdm
import time as systime
import numpy as np
import pandas as pd
import multiprocessing
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def main(path_prepared):
    initial_time = systime.time()
    print(f'Available cores for parallel processing: {multiprocessing.cpu_count()}')

    # Downsample current features
    print('Downsampling current features...')
    train_data = pd.read_hdf(path_prepared + 'SafeBaselines/current_features_train.h5', key='features')
    val_data = pd.read_hdf(path_prepared + 'SafeBaselines/current_features_val.h5', key='features')
    test_data = pd.read_hdf(path_prepared + 'SafeBaselines/current_features_test.h5', key='features')
    train_scene_ids = train_data['scene_id'].unique()
    val_scene_ids = val_data['scene_id'].unique()
    test_scene_ids = test_data['scene_id'].unique()

    variables = ['v_ego','v_sur','delta_v','psi_sur','acc_ego',
                 'v_ego2','v_sur2','delta_v2','rho']
    train_X = train_data[variables].values # (scenes, features#=9)
    assert train_X.shape[0] == len(train_scene_ids)
    val_X = val_data[variables].values
    assert val_X.shape[0] == len(val_scene_ids)
    test_X = test_data[variables].values
    assert test_X.shape[0] == len(test_scene_ids)
    full_X = np.concatenate([train_X, val_X, test_X], axis=0)

    scaler = StandardScaler()
    scaler.fit(full_X)

    while train_data['scene_id'].nunique() > 120000:
        train_scene_ids = train_data['scene_id'].unique()
        val_scene_ids = val_data['scene_id'].unique()

        train_X = train_data[variables].values # (scenes, features)
        val_X = val_data[variables].values
        full_X = scaler.transform(np.concatenate([train_X, val_X], axis=0))

        # k-Nearst Neighbors, k=100
        print('Setting up k-Nearest Neighbors search...')
        k = 100
        kNN = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='ball_tree', n_jobs=-1)
        kNN = kNN.fit(full_X)

        # Filter samples
        # 1. Compute the average distance to the k-th nearest neighbor;
        #    the distance is inversely proportional to the density of the samples
        # 2. Select samples with a probability inversely proportional to density, i.e., proportional to average distance
        #    let p = avg_distance / np.percentile(avg_distance, 75), to make sure the most infrequent 25% samples are reserved
        #    if a random number in (0,1) is less than p, the sample is reserved 

        avg_distances = np.zeros(full_X.shape[0]).astype(np.float32)
        chunck_size = 500
        for chunck_id in tqdm(range(0, full_X.shape[0], chunck_size), desc='Computing average distances', ascii=True, dynamic_ncols=False):
            chunck = full_X[chunck_id:chunck_id+chunck_size]
            distances, _ = kNN.kneighbors(chunck)
            avg_distances[chunck_id:chunck_id+chunck_size] = np.mean(distances[:,1:], axis=1)

        prob = avg_distances/(np.percentile(avg_distances,75))
        rand_uniform = np.random.RandomState(train_data['scene_id'].nunique()).uniform(size=len(avg_distances))

        train_selected_indices = np.where(rand_uniform[:train_X.shape[0]]<prob[:train_X.shape[0]])[0]
        val_selected_indices = np.where(rand_uniform[train_X.shape[0]:]<prob[train_X.shape[0]:])[0]
        print(f'{len(train_selected_indices)} training samples selected out of {train_X.shape[0]} samples, accounting for {len(train_selected_indices)/train_X.shape[0]*100:.2f}%')
        print(f'{len(val_selected_indices)} val samples selected out of {val_X.shape[0]} samples, accounting for {len(val_selected_indices)/val_X.shape[0]*100:.2f}%')

        train_data = train_data[train_data['scene_id'].isin(train_scene_ids[train_selected_indices])]
        val_data = val_data[val_data['scene_id'].isin(val_scene_ids[val_selected_indices])]

    train_data.to_hdf(path_prepared + 'SafeBaselines/current_features_train_downsampled.h5', key='features')
    val_data.to_hdf(path_prepared + 'SafeBaselines/current_features_val_downsampled.h5', key='features')

    # Generate "downsampled" environment features
    print('Generating downsampled environment features...')
    events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_table.csv').set_index('eventID')
    environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
    categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    events.loc[events['surfaceCondition']=='Other','surfaceCondition'] = 'Unknown'
    data2fit = events[environment_feature_names].fillna('Unknown')
    data2fit = data2fit.loc[(data2fit!='Unknown').all(axis=1)]
    categorical_encoder.fit(data2fit.values)

    categories_list = categorical_encoder.categories_
    categories_list = [list(categories)+['Unknown'] for categories in categories_list]
    all_combinations = np.array(list(product(*categories_list)))
    all_combinations = categorical_encoder.transform(all_combinations)
    all_combinations = pd.DataFrame(all_combinations, columns=categorical_encoder.get_feature_names_out(environment_feature_names))
    all_combinations = all_combinations.astype(np.int32)

    train_data = all_combinations.sample(frac=0.8, random_state=manual_seed)
    val_data = all_combinations.drop(train_data.index)
    train_data.to_hdf(path_prepared + 'SafeBaselines/environment_features_train_downsampled.h5', key='features')
    val_data.to_hdf(path_prepared + 'SafeBaselines/environment_features_val_downsampled.h5', key='features')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    main(path_prepared='./PreparedData/')
