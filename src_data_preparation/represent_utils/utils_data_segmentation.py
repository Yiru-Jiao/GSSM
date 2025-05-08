'''
This script contains utilities for segmenting data into scenes and organizing the features for model training.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from represent_utils.coortrans import coortrans


def read_dataset(dataset, path_processed, manual_seed):
    '''
    Read the selected dataset and organize the data into a single dataframe.
    '''
    if dataset=='highD':
        data_both = pd.concat([pd.read_hdf(path_processed+'highD/lane_changing/lc_0'+str(loc_id)+'.h5', key='data') for loc_id in range(1,7)], ignore_index=True)
        data_both['event_id'] = data_both['track_id_ego']
        data_both['target_id'] = data_both['track_id_ego'].astype(str) + '_' + data_both['track_id_sur'].astype(str)
    elif dataset=='SafeBaseline':
        data_ego = pd.concat([pd.read_hdf(path_processed+'SHRP2/SafeBaseline/Ego_birdseye_'+str(chunck_id)+'.h5', key='data') for chunck_id in range(0,5)], ignore_index=True)
        data_sur = pd.concat([pd.read_hdf(path_processed+'SHRP2/SafeBaseline/Surrounding_birdseye_'+str(chunck_id)+'.h5', key='data') for chunck_id in range(0,5)], ignore_index=True)
        data_ego['hx'] = np.cos(data_ego['psi_ekf'])
        data_ego['hy'] = np.sin(data_ego['psi_ekf'])
        data_sur['hx'] = np.cos(data_sur['psi_ekf'])
        data_sur['hy'] = np.sin(data_sur['psi_ekf'])
        data_ego = data_ego[['time','event_id','x_ekf','y_ekf','v_ekf','acc_ekf','hx','hy']]
        data_ego = data_ego.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v','acc_ekf':'acc_ego'})
        data_ego['vx'] = data_ego['v']*data_ego['hx']
        data_ego['vy'] = data_ego['v']*data_ego['hy']
        data_sur = data_sur[['time','event_id','target_id','x_ekf','y_ekf','v_ekf','hx','hy']]
        data_sur = data_sur.rename(columns={'x_ekf':'x','y_ekf':'y','v_ekf':'v'})
        data_sur['vx'] = data_sur['v']*data_sur['hx']
        data_sur['vy'] = data_sur['v']*data_sur['hy']

        data_both = data_ego.merge(data_sur, on=['event_id','time'], how='inner', suffixes=('_ego', '_sur'))
        meta_both = pd.read_csv(path_processed + 'SHRP2/metadata_birdseye.csv').set_index('event_id')
        veh_dimensions = meta_both[['ego_length','ego_width','target_length','target_width']].copy()
        veh_dimensions.loc[veh_dimensions['ego_length'].isna(), 'ego_length'] = np.nanmean(veh_dimensions['ego_length'].values)
        veh_dimensions.loc[veh_dimensions['ego_width'].isna(), 'ego_width'] = np.nanmean(veh_dimensions['ego_width'].values)
        data_both[['length_ego','width_ego']] = veh_dimensions.loc[data_both['event_id'].values, ['ego_length','ego_width']].values
        veh_dimensions = veh_dimensions[~veh_dimensions['target_length'].isna()].reset_index()
        random_dimensions = np.random.RandomState(manual_seed).choice(veh_dimensions.index, data_both['event_id'].nunique(), replace=True)
        random_dimensions = pd.DataFrame(veh_dimensions.loc[random_dimensions, ['target_length','target_width']].values,
                                         columns=['random_length','random_width'], index=data_both['event_id'].unique())
        data_both[['length_sur','width_sur']] = random_dimensions.loc[data_both['event_id'].values, ['random_length','random_width']].values
    elif dataset=='ArgoverseHV':
        data_both = pd.read_hdf(path_processed+'Argoverse/argo_hv.h5', key='data')
        data_both['event_id'] = data_both['log_id']
    return data_both


class ContextSegmenter(coortrans):
    '''
    This class segments the data into scenes and organizes the features for model training.
    '''
    def __init__(self, data, initial_scene_id, dataset, manual_seed):
        super().__init__()
        self.target_ids = data['target_id'].unique()
        data = data.sort_values(['target_id','time']).set_index(['target_id','time'])
        self.data = data
        self.initial_scene_id = initial_scene_id
        self.dataset = dataset
        self.manual_seed = manual_seed
        self.current_feature_size = 14
        if self.dataset=='SafeBaseline':
            self.events = pd.read_csv('./RawData/SHRP2/FileToUse/InsightTables/Event_Table.csv').set_index('eventID')
            self.environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
            self.one_hot_encoder = self.create_categorical_encoder(self.environment_feature_names)
        context_set = self.segment_data()
        if self.dataset=='SafeBaseline':
            self.profiles_set, self.current_features_set, self.environment_features_set = context_set
        else:
            self.profiles_set, self.current_features_set = context_set
    
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
        random_ends = np.random.RandomState(self.manual_seed).randint(1, 6, size=len(self.target_ids))

        for id_count, target_id in tqdm(enumerate(self.target_ids), desc='Target', total=len(self.target_ids), ascii=True, dynamic_ncols=False, miniters=100):
            df = self.data.loc(axis=0)[target_id, :].reset_index()
            if len(df)<35: # skip if the target was detected for less than 3.5 seconds
                continue
            else:
                event_id = df['event_id'].iloc[0]

            df_view_ego = self.transform_coor(df, 'ego')
            df_view_relative = self.transform_coor(df, 'relative')
            if self.dataset=='highD':
                indices_end = np.arange(len(df)-random_ends[id_count], 25, -8)
            elif self.dataset=='SafeBaseline':
                indices_end = np.arange(len(df)-random_ends[id_count], 25, -10)
            elif self.dataset=='ArgoverseHV':
                indices_end = np.arange(len(df)-random_ends[id_count], 25, -25)

            for idx_end in indices_end:
                # sample 2.5-second scenes
                df['psi_ego'] = self.angle(0, 1, df['hx_ego'], df['hy_ego']) # [-pi, pi]
                df['yaw_ego'] = np.gradient(np.unwrap(df['psi_ego']), df['time'])
                profiles = df.iloc[idx_end-25:idx_end][['target_id','time','yaw_ego','v_ego']]
                profiles['v_ego'] = abs(profiles['v_ego'])
                profiles['vx_sur'] = df_view_ego.iloc[idx_end-25:idx_end]['vx_sur'].values
                profiles['vy_sur'] = df_view_ego.iloc[idx_end-25:idx_end]['vy_sur'].values

                # skip if there is missing value in the profiles or df
                if profiles.isna().sum().sum()>0 or df.iloc[idx_end].isna().sum()>0:
                    continue
                profiles['scene_id'] = scene_id
                profiles = profiles.set_index(['target_id','time'])

                current_features = np.zeros(self.current_feature_size+1)
                vx_ego, vy_ego, vx_sur, vy_sur = df.iloc[idx_end][['vx_ego','vy_ego','vx_sur','vy_sur']].values

                current_features[0] = df.iloc[idx_end]['length_ego']
                current_features[1] = df.iloc[idx_end]['length_sur']
                current_features[2] = (df.iloc[idx_end]['width_ego']+df.iloc[idx_end]['width_sur'])/2
                current_features[3] = df_view_ego.iloc[idx_end]['vy_ego']
                current_features[4] = df_view_ego.iloc[idx_end]['vx_sur']
                current_features[5] = df_view_ego.iloc[idx_end]['vy_sur']
                current_features[6] = vx_ego**2 + vy_ego**2
                current_features[7] = vx_sur**2 + vy_sur**2
                current_features[8] = (vx_ego-vx_sur)**2 + (vy_ego-vy_sur)**2 # squared relative velocity
                current_features[9] = np.sqrt(current_features[8]) * np.sign(current_features[6]-current_features[7]) # relative speed
                current_features[10] = self.angle(0, 1, df_view_ego.iloc[idx_end]['hx_sur'], df_view_ego.iloc[idx_end]['hy_sur']) # heading angle of the surrounding vehicle
                current_features[11] = df.iloc[idx_end]['acc_ego']
                current_features[12] = self.angle(1, 0, df_view_relative.iloc[idx_end]['x_sur'], df_view_relative.iloc[idx_end]['y_sur']) # relative angle
                current_features[13] = np.sqrt(df_view_relative.iloc[idx_end]['x_sur']**2 + df_view_relative.iloc[idx_end]['y_sur']**2) # spacing
                current_features[-1] = scene_id

                # double check if there is any missing value
                if np.any(np.isnan(current_features)):
                    continue

                if id_count%10000==9999: # concat per every 10000 targets to speed up
                    profiles_set = [pd.concat(profiles_set, axis=0), profiles]
                else:
                    profiles_set.append(profiles)
                current_features_set.append(current_features)
                event_id_list.append([event_id, target_id, scene_id])
                scene_id += 1
        profiles_set = pd.concat(profiles_set, axis=0)
        profiles_set['scene_id'] = profiles_set['scene_id'].astype(int)
        current_features_set = pd.DataFrame(current_features_set, columns=['l_ego','l_sur','combined_width',
                                                                           'vy_ego','vx_sur','vy_sur','v_ego2','v_sur2','delta_v2','delta_v',
                                                                           'psi_sur','acc_ego','rho',
                                                                           's','scene_id'])
        current_features_set['scene_id'] = current_features_set['scene_id'].astype(int)
        event_id_list = np.array(event_id_list)
        if self.dataset=='SafeBaseline':
            environment_features_set = self.events.loc[event_id_list[:,0], self.environment_feature_names].fillna('Unknown')
            environment_features_set = self.one_hot_encoder.transform(environment_features_set.values)
            environment_features_set = pd.DataFrame(environment_features_set, columns=self.one_hot_encoder.get_feature_names_out(self.environment_feature_names))
            environment_features_set['event_id'] = event_id_list[:,0].astype(int)
            environment_features_set['target_id'] = event_id_list[:,1].astype(int)
            environment_features_set['scene_id'] = event_id_list[:,2].astype(int)
            return profiles_set, current_features_set, environment_features_set
        else:
            return profiles_set, current_features_set
