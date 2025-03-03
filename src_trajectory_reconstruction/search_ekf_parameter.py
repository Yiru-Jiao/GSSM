'''
This script contains functions to search for the optimal parameters for 
the Extended Kalman Filter (EKF) for ego and surrounding vehicle reconstruction.
'''

import os
import sys
import argparse
from tqdm import tqdm
import time as systime
import pandas as pd
import numpy as np
from reconstruction_utils.utils_ego_sur import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

path_raw = './RawData/SHRP2/'
path_raw_honda = './RawData/SHRP2/HondaDataSupport/'
path_raw_das = './RawData/SHRP2/DriverAssistanceSystems/'
path_processed = './ProcessedData/SHRP2/'

manual_seed = 131

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=4, help='The number of parallel jobs to run for grid search (defaults to -1 for all available cores)')
    parser.add_argument('--verbose', type=int, default=0, help='The verbosity level: 0 = silent')
    args = parser.parse_args()
    return args


# Define trainer
class trainer():
    def __init__(self, ego_uncertainty_init=100., ego_uncertainty_speed=100., ego_uncertainty_omega=10., ego_uncertainty_acc=10., 
                 ego_max_jerk=0.5, ego_max_yaw_rate=0.1, ego_max_acc=9.8, ego_max_yaw_acc=1.,
                 sur_uncertainty_init=15., sur_uncertainty_pos=2., sur_uncertainty_speed=8., 
                 sur_max_acc=9.8, sur_max_yaw_rate=0.5):
        self.ego_uncertainty_init = ego_uncertainty_init
        self.ego_uncertainty_speed = ego_uncertainty_speed
        self.ego_uncertainty_omega = ego_uncertainty_omega
        self.ego_uncertainty_acc = ego_uncertainty_acc
        self.ego_max_jerk = ego_max_jerk
        self.ego_max_yaw_rate = ego_max_yaw_rate
        self.ego_max_acc = ego_max_acc
        self.ego_max_yaw_acc = ego_max_yaw_acc
        self.sur_uncertainty_init = sur_uncertainty_init
        self.sur_uncertainty_pos = sur_uncertainty_pos
        self.sur_uncertainty_speed = sur_uncertainty_speed
        self.sur_max_acc = sur_max_acc
        self.sur_max_yaw_rate = sur_max_yaw_rate

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def apply_ekf(self, data):
        ego_params = dict(
            uncertainty_init=self.ego_uncertainty_init,
            uncertainty_speed=self.ego_uncertainty_speed,
            uncertainty_omega=self.ego_uncertainty_omega,
            uncertainty_acc=self.ego_uncertainty_acc,
            max_jerk=self.ego_max_jerk,
            max_yaw_rate=self.ego_max_yaw_rate,
            max_acc=self.ego_max_acc,
            max_yaw_acc=self.ego_max_yaw_acc)
        if self.target_veh=='ego':
            reconstructed_ego = []
            if self.verbose<2:
                progress_bar = data
            else:
                progress_bar = tqdm(data, total=len(data), desc='Applying EKF', dynamic_ncols=False, ascii=True, miniters=len(data)//100)
            for sample in progress_bar:
                df_ego, df_sur, ego_length = sample
                ## reconstruct ego trajectory
                df_ego = process_ego(df_ego, df_ego['event_id'].values[0], ego_params=ego_params)
                reconstructed_ego.append(df_ego)
            reconstructed_ego = pd.concat(reconstructed_ego)
            reconstructed_sur = pd.DataFrame()
        else:
            reconstructed_ego = []
            reconstructed_sur = []
            sur_params = dict(
                uncertainty_init=self.sur_uncertainty_init,
                uncertainty_pos=self.sur_uncertainty_pos,
                uncertainty_speed=self.sur_uncertainty_speed,
                max_acc=self.sur_max_acc,
                max_yaw_rate=self.sur_max_yaw_rate)
            if self.verbose<2:
                progress_bar = data
            else:
                progress_bar = tqdm(data, total=len(data), desc='Applying EKF', dynamic_ncols=False, ascii=True, miniters=len(data)//100)
            for sample in progress_bar:
                df_ego, df_sur, ego_length = sample
                ## reconstruct ego trajectory
                df_ego = process_ego(df_ego, df_ego['event_id'].values[0], ego_params=ego_params)
                reconstructed_ego.append(df_ego)
                ## reconstruct surrounding vehicle
                df_sur = process_surrounding(df_ego, df_sur, ego_length, sur_params=sur_params, n_jobs=1)
                reconstructed_sur.append(df_sur)
            reconstructed_ego = pd.concat(reconstructed_ego)
            reconstructed_sur = pd.concat(reconstructed_sur)
        return reconstructed_ego, reconstructed_sur

    def fit(self, train_data, target_veh, verbose):
        self.target_veh = target_veh
        self.verbose = verbose
        self.train_ego, self.train_sur = self.apply_ekf(train_data)
        return self

    def get_params(self, deep=False):
        return dict(
            ego_uncertainty_init=self.ego_uncertainty_init,
            ego_uncertainty_speed=self.ego_uncertainty_speed,
            ego_uncertainty_omega=self.ego_uncertainty_omega,
            ego_uncertainty_acc=self.ego_uncertainty_acc,
            ego_max_jerk=self.ego_max_jerk,
            ego_max_yaw_rate=self.ego_max_yaw_rate,
            ego_max_acc=self.ego_max_acc,
            ego_max_yaw_acc=self.ego_max_yaw_acc,
            sur_uncertainty_init=self.sur_uncertainty_init,
            sur_uncertainty_pos=self.sur_uncertainty_pos,
            sur_uncertainty_speed=self.sur_uncertainty_speed,
            sur_max_acc=self.sur_max_acc,
            sur_max_yaw_rate=self.sur_max_yaw_rate
        )

    def calculate_rmse(self, data, var_true, var_pred, scale_min, scale_max):
        series_true = data[var_true].values
        series_pred = data[var_pred].values
        reserve_idx = np.where(np.logical_not(np.isnan(series_true)))[0]
        series_true = series_true[reserve_idx]
        series_pred = series_pred[reserve_idx]
        series_true = (series_true - scale_min) / (scale_max - scale_min)
        series_pred = (series_pred - scale_min) / (scale_max - scale_min)
        return np.sqrt(np.mean((series_true - series_pred) ** 2))

    def score(self, test_data):
        self.test_ego, self.test_sur = self.apply_ekf(test_data)
        self.data_ego = pd.concat([self.train_ego, self.test_ego])
        self.data_sur = pd.concat([self.train_sur, self.test_sur])
        if self.target_veh=='ego':
            # ego speed rmse
            scale_min = 0.
            scale_max = 10.
            ego_speed_rmse = self.calculate_rmse(self.data_ego, 'speed_comp', 'v_ekf', scale_min, scale_max)
            # ego yaw rate rmse
            scale_min = -np.pi/12
            scale_max = np.pi/12
            ego_yaw_rate_rmse = self.calculate_rmse(self.data_ego, 'yaw_rate', 'omega_ekf', scale_min, scale_max)
            # ego acceleration rmse within max_acc
            scale_min = -3.
            scale_max = 3.
            ego_acc_rmse = self.calculate_rmse(self.data_ego, 'acc_lon', 'acc_ekf', scale_min, scale_max)
            ego_score = ego_speed_rmse+ego_yaw_rate_rmse+ego_acc_rmse
            if self.verbose>=1:
                sys.stderr.write(f'ego_score: {ego_score:.4f}\n')
            return -ego_score
        else:
            # sur (x, y) mean displacement error
            scale_min = 0.
            scale_max = 5.
            displacement = np.sqrt((self.data_sur['x_ekf'].values-self.data_sur['x'].values)**2 +
                                   (self.data_sur['y_ekf'].values-self.data_sur['y'].values)**2)
            displacement = displacement[~np.isnan(displacement)]
            displacement = (displacement - scale_min) / (scale_max - scale_min)
            displacement_error = np.mean(displacement)
            # sur speed rmse
            scale_min = 0.
            scale_max = 10.
            sur_speed_rmse = self.calculate_rmse(self.data_sur, 'speed_comp', 'v_ekf', scale_min, scale_max)
            sur_score = displacement_error+sur_speed_rmse
            if self.verbose>=1:
                sys.stderr.write(f'sur_score: {sur_score:.4f}\n')
            return -sur_score


# Define grid search function
def grid_search(param_space, dataset, target_veh, n_jobs, verbose):
    '''
    The search will use all input data, no need to split into train and test set
    This split is only for adapting the grid search function
    '''
    zero_fold = ShuffleSplit(n_splits=1, test_size=0.2, random_state=manual_seed)

    scorer = trainer()
    gs = GridSearchCV(scorer, param_space, cv=zero_fold, n_jobs=n_jobs, verbose=max(0,verbose-1), refit=False)
    gs.fit(dataset, **{'target_veh': target_veh, 'verbose': verbose})
    best_params, best_score = gs.best_params_, round(gs.best_score_, 4)

    del scorer
    del gs
    
    return best_params, best_score


def main(args):
    initial_time = systime.time()

    # Load metadata and event information
    meta_both = pd.read_csv(path_processed + 'metadata_birdseye.csv')
    events = pd.read_csv(path_raw_honda + 'InsightTables_csv/Event_Table.csv')
    meta_both = meta_both.set_index('event_id')
    events = events.set_index('eventID')

    # Sample 100 events for each event category for parameter search
    if not os.path.exists(path_raw + 'ego_length_ekf_param_search.csv'):
        data_ego = []
        data_sur = []
        ego_length_list = []
        target_id = 0 # Initialize target_id for surrounding vehicles detected by radar
        for event_cat in meta_both['event_category'].value_counts().index.values:
            sample_count = 0
            if event_cat=='SafeBaseline':
                sample_number = 500
            else:
                sample_number = 100
            event_ids = meta_both[(meta_both['event_category']==event_cat)&
                                (~meta_both['file2use'].isna())].index.sort_values().values
            
            progress_bar = tqdm(event_ids, desc=event_cat)
            for event_id in progress_bar:
                ## read time series data
                try:
                    sample = pd.read_csv(meta_both.loc[event_id]['file_dir'] + meta_both.loc[event_id]['file2use'], on_bad_lines='warn')
                except:
                    # print(f'Error reading {event_id}, skip')
                    continue

                ## check if the event has nan values
                continue_flag = False
                for var in ['vtti.speed_network','vtti.accel_x','vtti.accel_y','vtti.gyro_z']:
                    if sample[var].isna().sum() > 0.2*len(sample):
                        # print(f'{event_id} has more than 20% nan {var}, skip')
                        continue_flag = True
                        break
                if continue_flag:
                    continue

                ## create dataframe (note: SHRP2 excluded rearward vehicles)
                df_ego, df_forward, target_id, _ = create_dataframe(sample, event_id, target_id)
                if np.all(df_ego['speed_comp']<=1e-6):
                    # print(f'{event_id} has all zero ego speed, skip')
                    continue
                valid_start = np.all(df_ego['speed_comp'].iloc[:5]>=0)
                valid_end = np.all(df_ego['speed_comp'].iloc[-5:]>=0)
                if not valid_start|valid_end:
                    # print(f'{event_id} start&end speed not available, skip')
                    continue
                if len(df_forward)>0:
                    ego_length = meta_both.loc[event_id]['ego_length']
                    if np.isnan(ego_length):
                        # print(f'{event_id} ego length not available, skip')
                        continue
                    else:
                        max_sur_duration = df_forward.groupby('target_id')['time'].size().max()
                        if max_sur_duration<20:
                            # print(f'{event_id} has too short surrounding vehicle duration, skip')
                            continue

                        continue_flag = False
                        for var in ['local_dx','local_dy','delta_vx','delta_vy']:
                            if df_forward[var].isna().sum() > 0.2*len(df_forward):
                                # print(f'{event_id} has more than 20% nan in surrounding vehicle {var}, skip')
                                continue_flag = True
                                break
                        if continue_flag:
                            continue
                else:
                    # print(f'{event_id} has no surrounding vehicle, skip')
                    continue
                
                data_ego.append(df_ego)
                data_sur.append(df_forward)
                ego_length_list.append([event_id, ego_length])

                sample_count += 1
                progress_bar.set_postfix(sample_count=sample_count)
                if sample_count >= sample_number:
                    break

        print(f'In total {len(data_ego)} samples for parameter search.')
        data_ego = pd.concat(data_ego)
        data_sur = pd.concat(data_sur)
        ego_length_list = pd.DataFrame(ego_length_list, columns=['event_id', 'ego_length'])
        data_ego.to_hdf(path_raw + 'samples_ego_ekf_param_search.h5', key='data_ego', mode='w')
        data_sur.to_hdf(path_raw + 'samples_sur_ekf_param_search.h5', key='data_sur', mode='w')
        ego_length_list.to_csv(path_raw + 'ego_length_ekf_param_search.csv', index=False)


    # Read and organise dataset
    print('Reading dataset...')
    data_ego = pd.read_hdf(path_raw + 'samples_ego_ekf_param_search.h5', key='data_ego')
    data_sur = pd.read_hdf(path_raw + 'samples_sur_ekf_param_search.h5', key='data_sur')
    ego_length_list = pd.read_csv(path_raw + 'ego_length_ekf_param_search.csv')
    dataset = []
    for event_id, ego_length in ego_length_list.values:
        df_ego = data_ego[data_ego['event_id']==event_id].sort_values('time')
        df_sur = data_sur[data_sur['event_id']==event_id].sort_values(['target_id','time'])
        dataset.append([df_ego, df_sur, ego_length])


    # Grid search
    cpu_number = os.cpu_count()
    if cpu_number<32:
        '''
        With 64 CPUs, the search takes around 11 hours
        '''
        print(f'Number of CPUs: {cpu_number}, may not be efficient enough for grid search.')
        sys.exit(0)
    else:
        print(f'Number of CPUs: {cpu_number}, starting grid search...')
     
    if os.path.exists(path_processed + 'ekf_parameters.csv'):
        result = pd.read_csv(path_processed + 'ekf_parameters.csv')
        if 'ego_uncertainty_init' in result.columns:
            ego_trained = True
        else:
            ego_trained = False
        if 'sur_uncertainty_init' in result.columns:
            if result['sur_uncertainty_init'].values[0]>0:
                sur_trained = True
            else:
                sur_trained = False
        else:
            sur_trained = False
    else:
        ego_trained = False
        sur_trained = False

    if ego_trained & sur_trained:
        print('EKF parameters already trained.')
        sys.exit(0)

    if not ego_trained:
        ## Ego only
        search_space_ego = {'ego_uncertainty_init': [16., 32., 64., 128.],
                            'ego_uncertainty_speed': [2., 4., 8., 16., 32.],
                            'ego_uncertainty_omega': [0.01, 0.05, 0.1, 0.25],
                            'ego_uncertainty_acc': [2., 4., 8., 16., 32.],
                            'ego_max_jerk': [2., 4., 8., 16.],
                            'ego_max_yaw_rate': [np.pi/15, np.pi/10, np.pi/5, np.pi/2],
                            'ego_max_acc': [6.5, 9.8],
                            'ego_max_yaw_acc': [np.pi/5, np.pi/2, np.pi, 2*np.pi],
                            'sur_uncertainty_init': [np.nan],
                            'sur_uncertainty_pos': [np.nan],
                            'sur_uncertainty_speed': [np.nan],
                            'sur_max_acc': [np.nan],
                            'sur_max_yaw_rate': [np.nan]}

        params_ego, score_ego = grid_search(search_space_ego, dataset, target_veh='ego', n_jobs=args.n_jobs, verbose=args.verbose)
        result = pd.DataFrame(params_ego, index=[0])
        result['score_ego'] = score_ego
        result.to_csv(path_processed + 'ekf_parameters.csv', index=False)
    else:
        params_ego = result.iloc[0].to_dict()

    if not sur_trained:
        ## Ego and surrounding vehicles
        search_space_sur = {'sur_uncertainty_init': [16., 32., 64., 128.],
                            'sur_uncertainty_pos': [2., 4., 8., 16., 32.],
                            'sur_uncertainty_speed': [2., 4., 8., 16., 32.],
                            'sur_max_acc': [6.5, 9.8],
                            'sur_max_yaw_rate': [np.pi/5, np.pi/2, np.pi, 2*np.pi],
                            'ego_uncertainty_init': [params_ego['ego_uncertainty_init']],
                            'ego_uncertainty_speed': [params_ego['ego_uncertainty_speed']],
                            'ego_uncertainty_omega': [params_ego['ego_uncertainty_omega']],
                            'ego_uncertainty_acc': [params_ego['ego_uncertainty_acc']],
                            'ego_max_jerk': [params_ego['ego_max_jerk']],
                            'ego_max_yaw_rate': [params_ego['ego_max_yaw_rate']],
                            'ego_max_acc': [params_ego['ego_max_acc']],
                            'ego_max_yaw_acc': [params_ego['ego_max_yaw_acc']]}

        params_sur, score_sur = grid_search(search_space_sur, dataset, target_veh='sur', n_jobs=args.n_jobs, verbose=args.verbose)

        result['sur_uncertainty_init'] = params_sur['sur_uncertainty_init']
        result['sur_uncertainty_pos'] = params_sur['sur_uncertainty_pos']
        result['sur_uncertainty_speed'] = params_sur['sur_uncertainty_speed']
        result['sur_max_acc'] = params_sur['sur_max_acc']
        result['sur_max_yaw_rate'] = params_sur['sur_max_yaw_rate']
        result['score_sur'] = score_sur

        result.to_csv(path_processed + 'ekf_parameters.csv', index=False)
        
    print('--- Time elapsed in total : ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time()-initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
    