'''
This script segments data into scenes and organises the features for model training.
'''

import os
import sys
import multiprocessing
import time as systime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
font = {'family' : 'Arial',
        'size'   : 9}
plt.rc('font', **font)
from matplotlib.ticker import ScalarFormatter
from represent_utils.utils_data_segmentation import TimeSeriesSegmenter


def main(path_prepared, path_processed):
    initial_time = systime.time()
    print(f'Available cores for parallel processing: {multiprocessing.cpu_count()}')

    for safe_or_conflict in ['Safe', 'Conflict']:
        if safe_or_conflict=='Safe' and os.path.exists(path_prepared + 'Segments/profiles_safe.h5'):
            print('Safe set already segmented.')
            continue
        elif safe_or_conflict=='Conflict' and os.path.exists(path_prepared + 'Segments/profiles_train.h5'):
            print('Conflict set already segmented.')
            continue

        print('Loading data...')
        if safe_or_conflict=='Safe':
            data_ego = pd.concat([pd.read_hdf(path_processed+'SafeBaseline/Ego_birdseye_'+str(chunck_id)+'.h5', key='data') for chunck_id in range(0,5)], ignore_index=True)
            data_sur = pd.concat([pd.read_hdf(path_processed+'SafeBaseline/Surrounding_birdseye_'+str(chunck_id)+'.h5', key='data') for chunck_id in range(0,5)], ignore_index=True)
        else:
            conflict_folders = os.listdir(path_processed)
            conflict_folders = [folder for folder in conflict_folders if 'Safe' not in folder and os.path.isdir(path_processed+folder)]
            data_ego = pd.concat([pd.read_hdf(path_processed+folder+'/Ego_birdseye.h5', key='data') for folder in conflict_folders], ignore_index=True)
            data_sur = pd.concat([pd.read_hdf(path_processed+folder+'/Surrounding_birdseye.h5', key='data') for folder in conflict_folders], ignore_index=True)
        data_ego['hx'] = np.cos(data_ego['psi_ekf'])
        data_ego['hy'] = np.sin(data_ego['psi_ekf'])
        data_sur['hx'] = np.cos(data_sur['psi_ekf'])
        data_sur['hy'] = np.sin(data_sur['psi_ekf'])

        data_ego = data_ego[['time','event_id','x_ekf','y_ekf','v_ekf','omega_ekf','acc_ekf','hx','hy']]
        data_ego = data_ego.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v','omega_ekf':'omega_ego','acc_ekf':'acc_ego'})
        data_sur = data_sur[['time','event_id','target_id','x_ekf','y_ekf','v_ekf','hx','hy']]
        data_sur = data_sur.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v'})
        data_both = data_ego.merge(data_sur, on=['event_id','time'], how='inner', suffixes=('_ego', '_sur'))
        data_ego, data_sur = [], [] ## free memory

        meta_both = pd.read_csv(path_processed + 'metadata_birdseye.csv')
        meta_both = meta_both.set_index('event_id')
        veh_dimensions = meta_both[['ego_length','target_length','other_length']].copy()
        veh_dimensions.loc[veh_dimensions['ego_length'].isna(), 'ego_length'] = np.nanmean(veh_dimensions['ego_length'].values)
        no_target_length = (veh_dimensions['target_length'].isna())&(veh_dimensions['other_length'].isna())
        random_lengths = np.random.RandomState(manual_seed).choice(veh_dimensions['target_length'].dropna().values, no_target_length.sum(), replace=True)
        veh_dimensions.loc[no_target_length, 'target_length'] = random_lengths

        path_save = path_prepared + 'Segments/'
        os.makedirs(path_save, exist_ok=True)
        # Segment and save scenes, with profiles and current features separated
        if safe_or_conflict=='Safe':
            event_ids = data_both['event_id'].unique()
            # 584,985 scenes in total
            initial_scene_id = 500000
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.), constrained_layout=True)
            bins = np.linspace(0, 40, 31)
            print('Segmenting safe set...')
            sr = TimeSeriesSegmenter(data_both, veh_dimensions, initial_scene_id)
            sr.profiles_set.to_hdf(path_save + 'profiles_safe.h5', key='profiles')
            sr.current_features_set.to_hdf(path_save + 'current_features_safe.h5', key='features')
            sr.environment_features_set.to_hdf(path_save + 'environment_features_safe.h5', key='features')
            print('Number of scenes in safe set: ' + str(sr.current_features_set['scene_id'].nunique()))
            print(f'Minimum net distance: {sr.current_features_set['s'].min():.2f}')
            print(f'Unique scene ids in current features set: {sr.current_features_set['scene_id'].nunique()}, should be the same as the profiles set: {sr.profiles_set['scene_id'].nunique()}')
            '''
            In safe set: minimum net distance: 1.64m
            '''
            ## save a plot of speed distribution
            ax.hist(sr.profiles_set['v_ego'], bins=bins, alpha=0.5, label='Ego vehicle')
            ax.hist(sr.profiles_set['v_sur'], bins=bins, alpha=0.5, label='Surrounding vehicles')
            ax.set_xlabel('Speed (m/s)')
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax.set_ylabel('Frequency')
            ax.set_title('Safe set')
            ax.legend()
            fig.savefig(path_save + 'speed_distribution_safe.pdf', bbox_inches='tight', dpi=600)
        else:
            event_ids = data_both['event_id'].unique()
            len_event_ids = len(event_ids)
            # Separate all the events into train (70%, 177,650 scenes), val (10%, 25,514), test (20%, 50,908) sets
            val_event_ids = np.random.RandomState(manual_seed).choice(event_ids, int(0.1*len_event_ids), replace=False)
            event_ids = np.setdiff1d(event_ids, val_event_ids)
            test_event_ids = np.random.RandomState(manual_seed).choice(event_ids, int(0.2*len_event_ids), replace=False)
            train_event_ids = np.setdiff1d(event_ids, test_event_ids)

            data_train = data_both[data_both['event_id'].isin(train_event_ids)]
            data_val = data_both[data_both['event_id'].isin(val_event_ids)]
            data_test = data_both[data_both['event_id'].isin(test_event_ids)]

            # Segment and save scenes, with profiles and current features separated
            initial_scene_id = 0
            fig, axes = plt.subplots(1, 3, figsize=(8., 2.), constrained_layout=True)
            bins = np.linspace(0, 40, 31)
            for data, suffix in zip([data_train, data_val, data_test], ['train', 'val', 'test']):
                print('Segmenting ' + suffix + ' set...')
                sr = TimeSeriesSegmenter(data, veh_dimensions, initial_scene_id)
                sr.profiles_set.to_hdf(path_save + 'profiles_'+suffix+'.h5', key='profiles')
                sr.current_features_set.to_hdf(path_save + 'current_features_'+suffix+'.h5', key='features')
                sr.environment_features_set.to_hdf(path_save + 'environment_features_'+suffix+'.h5', key='features')
                initial_scene_id = sr.current_features_set['scene_id'].max() + 1
                print('Number of scenes in ' + suffix + ' set: ' + str(initial_scene_id - sr.initial_scene_id))
                print(f'Minimum net distance: {sr.current_features_set['s'].min():.2f}')
                print(f'Unique scene ids in current features set: {sr.current_features_set['scene_id'].nunique()}, should be the same as the profiles set: {sr.profiles_set['scene_id'].nunique()}')
                '''
                In train set: minimum net distance: 0.23m
                In val set: minimum net distance: 0.38m
                In test set: minimum net distance: 0.17m
                '''
                ## save a plot of speed distribution
                ax = axes[0] if suffix=='train' else axes[1] if suffix=='val' else axes[2]
                ax.hist(sr.profiles_set['v_ego'], bins=bins, alpha=0.5, label='Ego vehicle')
                ax.hist(sr.profiles_set['v_sur'], bins=bins, alpha=0.5, label='Surrounding vehicles')
                ax.set_xlabel('Speed (m/s)' if suffix=='train' or suffix=='test' else '')
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.set_ylabel('Frequency')
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
            axes[0].set_title('Train set')
            axes[1].set_title('Val set')
            axes[2].set_title('Test set')
            fig.savefig(path_save + 'speed_distribution.pdf', bbox_inches='tight', dpi=600)

    
    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    path_prepared = './PreparedData/'
    path_processed = './ProcessedData/'
    main(path_prepared, path_processed)
