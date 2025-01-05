'''
This script selects a subset of samples from the training and val sets, 
based on the density of the samples in the feature space.
The density is computed as the inverse of the average distance to k nearest neighbours.
The samples are selected with a probability inversely proportional to the density,
to ensure that the infrequent samples are selected.

This downsampling is only for representation learning to reduce "too similar" samples.
The complete dataset will be used for training the final model.

http://dx.doi.org/10.4236/jilsa.2015.74010
'''

import os
import sys
from tqdm import tqdm
import time as systime
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def main(path_prepared):
    initial_time = systime.time()
    print(f'Available cores for parallel processing: {multiprocessing.cpu_count()}')

    if os.path.exists(path_prepared + 'SafeBaselines/profiles_val_downsampled.h5'):
        print('downsampled data already exists. Skip downsampling and start plotting...')
        train_data = pd.read_hdf(path_prepared + 'SafeBaselines/profiles_train.h5', key='profiles')
        val_data = pd.read_hdf(path_prepared + 'SafeBaselines/profiles_val.h5', key='profiles')
        train_data_downsampled = pd.read_hdf(path_prepared + 'SafeBaselines/profiles_train_downsampled.h5', key='profiles')
        val_data_downsampled = pd.read_hdf(path_prepared + 'SafeBaselines/profiles_val_downsampled.h5', key='profiles')
        print(f"{train_data['scene_id'].nunique()} training samples selected.")
        print(f"{val_data['scene_id'].nunique()} val samples selected.")
    else:
        # Load segmented data
        print('Loading segmented data...')
        train_data = pd.read_hdf(path_prepared + 'SafeBaselines/profiles_train.h5', key='profiles')
        val_data = pd.read_hdf(path_prepared + 'SafeBaselines/profiles_val.h5', key='profiles')
        test_data = pd.read_hdf(path_prepared + 'SafeBaselines/profiles_test.h5', key='profiles')
        train_scene_ids = train_data['scene_id'].unique()
        val_scene_ids = val_data['scene_id'].unique()
        test_scene_ids = test_data['scene_id'].unique()

        train_X = train_data[['v_ego','omega_ego','v_sur']].values.reshape(-1,60) # (scenes, time_steps*features)
        assert train_X.shape[0] == len(train_scene_ids)
        val_X = val_data[['v_ego','omega_ego','v_sur']].values.reshape(-1,60)
        assert val_X.shape[0] == len(val_scene_ids)
        test_X = test_data[['v_ego','omega_ego','v_sur']].values.reshape(-1,60)
        assert test_X.shape[0] == len(test_scene_ids)
        full_X = np.concatenate([train_X, val_X, test_X], axis=0)

        scaler = StandardScaler()
        scaler.fit(full_X.reshape(-1, 3)) # features: v_ego, omega_ego, v_sur

        while train_data['scene_id'].nunique() > 200000:
            train_scene_ids = train_data['scene_id'].unique()
            val_scene_ids = val_data['scene_id'].unique()

            train_X = train_data[['v_ego','omega_ego','v_sur']].values.reshape(-1,60) # (scenes, time_steps*features)
            val_X = val_data[['v_ego','omega_ego','v_sur']].values.reshape(-1,60)
            full_X = np.concatenate([train_X, val_X], axis=0)
            full_X = scaler.transform(full_X.reshape(-1, 3)).reshape(full_X.shape)

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

        train_data.to_hdf(path_prepared + 'SafeBaselines/profiles_train_downsampled.h5', key='profiles')
        val_data.to_hdf(path_prepared + 'SafeBaselines/profiles_val_downsampled.h5', key='profiles')

    ## save a plot of speed distribution
    import matplotlib.pyplot as plt
    font = {'family' : 'Arial',
            'size'   : 9}
    plt.rc('font', **font)
    from matplotlib.ticker import ScalarFormatter
    fig, axes = plt.subplots(1, 3, figsize=(8., 1.8), constrained_layout=True)
    speed_bins = np.linspace(0, 40, 31)
    yaw_bins = np.linspace(-np.pi/40, np.pi/40, 31)
    for var, var_name in zip(['v_ego', 'omega_ego', 'v_sur'], ['Speed (m/s)', 'Yaw rate (rad/s)', 'Speed (m/s)']):
        ax = axes[0] if var=='v_ego' else axes[1] if var=='omega_ego' else axes[2]
        bins = speed_bins if var=='v_ego' or var=='v_sur' else yaw_bins
        ax.hist(train_data[var], bins=bins, alpha=0.5, label='Original dataset')
        ax.hist(train_data_downsampled[var], bins=bins, alpha=0.5, label='Downsampled dataset')
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylabel('Frequency')
        ax.set_title(var_name)
        ax.set_xlabel('')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.15))
    fig.savefig(path_prepared + 'SafeBaselines/speed_distribution_downsampled.pdf', bbox_inches='tight', dpi=600)

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    main(path_prepared='./PreparedData/')
