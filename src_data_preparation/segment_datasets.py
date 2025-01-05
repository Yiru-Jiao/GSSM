'''
'''

import os
import sys
import multiprocessing
import time as systime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
font = {'family' : 'Arial',
        'size'   : 9}
plt.rc('font', **font)
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import OneHotEncoder
from represent_utils.coortrans import coortrans


class TimeSeriesSegmenter(coortrans):
    def __init__(self, data, initial_scene_id):
        super().__init__()
        self.target_ids = data['target_id'].unique()
        data = data.sort_values(['target_id','time']).set_index(['target_id','time'])
        self.data = data
        self.current_feature_size = 10
        self.initial_scene_id = initial_scene_id
        self.events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_table.csv').set_index('eventID')
        self.environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
        self.one_hot_encoder = self.create_categorical_encoder(self.environment_feature_names)
        self.profiles_set, self.current_features_set, self.environment_features_set = self.segment_data()

    # Define a encoder for categorical variables
    def create_categorical_encoder(self, environment_feature_names):
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.events.loc[self.events['surfaceCondition']=='Other','surfaceCondition'] = 'Unknown'
        data2fit = self.events[environment_feature_names].fillna('Unknown')
        data2fit = data2fit.loc[(data2fit!='Unknown').all(axis=1)]
        categorical_encoder.fit(data2fit.values)
        print('Categories:', categorical_encoder.categories_)
        return categorical_encoder

    # Segment data and organize features
    def segment_data(self,):
        profiles_set = []
        current_features_set = []
        event_id_list = []
        scene_id = self.initial_scene_id

        for target_id in tqdm(self.target_ids, desc='Target'):
            df = self.data.loc(axis=0)[target_id, :]
            if len(df)<25: # skip if the target was detected for less than 2.5 seconds
                continue
            else:
                event_id = df['event_id'].iloc[0]

            df_view_ego = self.transform_coor(df, 'ego')
            df_view_relative = self.transform_coor(df, 'relative')
            indices_end = np.arange(len(df)-1, 23, -10) # avoid the first 0.3 second due to likely unreliable data
            for idx_end in indices_end:
                # sample 2-second scenes every 1 second
                profiles = df.iloc[idx_end-20:idx_end][['v_ego','omega_ego','v_sur']] # speed and yaw rate of ego, speed of surrounding

                # if there is no missing value in the profiles or df
                if profiles.isna().sum().sum()==0 and df.iloc[idx_end].isna().sum()==0:
                    profiles['scene_id'] = scene_id

                    current_features = np.zeros(self.current_feature_size+1)
                    current_features[:2] = df.iloc[idx_end][['v_ego','v_sur']]
                    vx_ego = df.iloc[idx_end]['v_ego']*df.iloc[idx_end]['hx_ego']
                    vy_ego = df.iloc[idx_end]['v_ego']*df.iloc[idx_end]['hy_ego']
                    vx_sur = df.iloc[idx_end]['v_sur']*df.iloc[idx_end]['hx_sur']
                    vy_sur = df.iloc[idx_end]['v_sur']*df.iloc[idx_end]['hy_sur']
                    current_features[2] = np.sqrt((vx_ego-vx_sur)**2 + (vy_ego-vy_sur)**2)
                    current_features[3] = self.angle(1, 0, df_view_ego.iloc[idx_end]['hx_sur'], df_view_ego.iloc[idx_end]['hy_sur'])
                    current_features[4] = df.iloc[idx_end]['acc_ego']
                    current_features[5] = df.iloc[idx_end]['v_ego']**2
                    current_features[6] = df.iloc[idx_end]['v_sur']**2
                    current_features[7] = current_features[2]**2
                    current_features[8] = self.angle(1, 0, df_view_relative.iloc[idx_end]['x_sur'], df_view_relative.iloc[idx_end]['y_sur'])
                    current_features[9] = np.sqrt(df_view_relative.iloc[idx_end]['x_sur']**2 + df_view_relative.iloc[idx_end]['y_sur']**2)
                    current_features[-1] = scene_id

                    profiles_set.append(profiles)
                    current_features_set.append(current_features)
                    event_id_list.append([event_id, target_id, scene_id])
                    scene_id += 1
        profiles_set = pd.concat(profiles_set, axis=0)
        profiles_set['scene_id'] = profiles_set['scene_id'].astype(int)
        current_features_set = pd.DataFrame(current_features_set, columns=['v_ego','v_sur','delta_v',
                                                                           'psi_sur','acc_ego',
                                                                           'v_ego2','v_sur2','delta_v2',
                                                                           'rho','s','scene_id'])
        current_features_set['scene_id'] = current_features_set['scene_id'].astype(int)
        event_id_list = np.array(event_id_list)
        environment_features_set = self.events.loc[event_id_list[:,0], self.environment_feature_names].fillna('Unknown')
        environment_features_set = self.one_hot_encoder.transform(environment_features_set.values)
        environment_features_set = pd.DataFrame(environment_features_set, columns=self.one_hot_encoder.get_feature_names_out(self.environment_feature_names))
        environment_features_set['event_id'] = event_id_list[:,0].astype(int)
        environment_features_set['target_id'] = event_id_list[:,1].astype(int)
        environment_features_set['scene_id'] = event_id_list[:,2].astype(int)
        return profiles_set, current_features_set, environment_features_set


def main(path_prepared, path_processed):
    initial_time = systime.time()
    print(f'Available cores for parallel processing: {multiprocessing.cpu_count()}')

    # Separate all the events into train (70%, 177,650 scenes), val (10%, 25,514), test (20%, 50,908) sets
    print('Loading data...')
    conflict_folders = os.listdir(path_processed)
    conflict_folders = [folder for folder in conflict_folders if 'Safe' not in folder and os.path.isdir(path_processed+folder)]

    data_ego = pd.concat([pd.read_hdf(path_processed+folder+'/Ego_birdseye.h5', key='data') for folder in conflict_folders], ignore_index=True)
    data_ego['hx'] = np.cos(data_ego['psi_ekf'])
    data_ego['hy'] = np.sin(data_ego['psi_ekf'])
    data_sur = pd.concat([pd.read_hdf(path_processed+folder+'/Surrounding_birdseye.h5', key='data') for folder in conflict_folders], ignore_index=True)
    data_sur['hx'] = np.cos(data_sur['psi_ekf'])
    data_sur['hy'] = np.sin(data_sur['psi_ekf'])

    data_ego = data_ego[['time','event_id','x_ekf','y_ekf','v_ekf','omega_ekf','acc_ekf','hx','hy']]
    data_ego = data_ego.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v','omega_ekf':'omega_ego','acc_ekf':'acc_ego'})
    data_sur = data_sur[['time','event_id','target_id','x_ekf','y_ekf','v_ekf','hx','hy']]
    data_sur = data_sur.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v'})
    data_both = data_ego.merge(data_sur, on=['event_id','time'], how='inner', suffixes=('_ego', '_sur'))
    data_ego, data_sur = [], [] ## free memory

    event_ids = data_both['event_id'].unique()
    len_event_ids = len(event_ids)
    val_event_ids = np.random.RandomState(manual_seed).choice(event_ids, int(0.1*len_event_ids), replace=False)
    event_ids = np.setdiff1d(event_ids, val_event_ids)
    test_event_ids = np.random.RandomState(manual_seed).choice(event_ids, int(0.2*len_event_ids), replace=False)
    train_event_ids = np.setdiff1d(event_ids, test_event_ids)

    data_train = data_both[data_both['event_id'].isin(train_event_ids)]
    data_val = data_both[data_both['event_id'].isin(val_event_ids)]
    data_test = data_both[data_both['event_id'].isin(test_event_ids)]


    # Segment and save scenes, with profiles and current features separated
    initial_scene_id = 0
    path_save = path_prepared + 'Segments/'
    os.makedirs(path_save, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8., 2.), constrained_layout=True)
    bins = np.linspace(0, 40, 31)
    for data, suffix in zip([data_train, data_val, data_test], ['train', 'val', 'test']):
        print('Segmenting ' + suffix + ' set...')
        sr = TimeSeriesSegmenter(data, initial_scene_id)
        sr.profiles_set.to_hdf(path_save + 'profiles_'+suffix+'.h5', key='profiles')
        sr.current_features_set.to_hdf(path_save + 'current_features_'+suffix+'.h5', key='features')
        sr.environment_features_set.to_hdf(path_save + 'environment_features_'+suffix+'.h5', key='features')
        initial_scene_id = sr.current_features_set['scene_id'].max() + 1
        print('Number of scenes in ' + suffix + ' set: ' + str(initial_scene_id - sr.initial_scene_id))
        print(f'Minimum net distance: {sr.current_features_set['s'].min():.2f}, minimum ego speed: {sr.current_features_set['v_ego'].min():.2f}')
        print(f'Unique scene ids in current features set: {sr.current_features_set['scene_id'].nunique()}, should be the same as the profiles set: {sr.profiles_set['scene_id'].nunique()}')
        '''
        In train set: minimum net distance: 0.23m, minimum ego speed: 0.00
        In val set: minimum net distance: 0.38m, minimum ego speed: 0.00
        In test set: minimum net distance: 0.17m, minimum ego speed: 0.00
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