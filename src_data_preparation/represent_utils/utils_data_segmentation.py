'''
This script contains utilities for segmenting data into scenes and organizing the features for model training.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from represent_utils.coortrans import coortrans


class TimeSeriesSegmenter(coortrans):
    def __init__(self, data, veh_dimensions, initial_scene_id):
        super().__init__()
        self.target_ids = data['target_id'].unique()
        data = data.sort_values(['target_id','time']).set_index(['target_id','time'])
        self.data = data
        self.veh_dimensions = veh_dimensions
        self.current_feature_size = 10
        self.initial_scene_id = initial_scene_id
        self.events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_Table.csv').set_index('eventID')
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

        for target_id in tqdm(self.target_ids, desc='Target', total=len(self.target_ids)):
            df = self.data.loc(axis=0)[target_id, :]
            if len(df)<25: # skip if the target was detected for less than 2.5 seconds
                continue
            else:
                event_id = df['event_id'].iloc[0]
            veh_dimension = self.veh_dimensions.loc[event_id, ['ego_length','target_length','other_length']].values
            ego_length, target_length, other_length = veh_dimension
            if np.isnan(target_length):
                target_length = other_length

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
                    current_features[0] = ego_length
                    current_features[1] = target_length
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
        current_features_set = pd.DataFrame(current_features_set, columns=['l_ego','l_sur','delta_v',
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

