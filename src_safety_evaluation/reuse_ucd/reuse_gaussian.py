'''
This script reuses the Unified Conflict Detection (UCD) model to evaluate the event safety.
'''

import os
import sys
import random
import time as systime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import argparse
sys.path.append('./')
from scipy.special import erf
from unified_conflit_detection import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from joblib import Parallel, delayed
from src_safety_evaluation.validation_utils.utils_evaluation import *



# Set input and output paths
path_input = './Data/InputData/'
path_output = './Data/OutputData/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    return args


def main(args, manual_seed, path_result):
    initial_time = systime.time()
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    
    # Set the random seed
    if args.reproduction:
        args.seed = manual_seed # Fix the random seed for reproduction
    if args.seed is None:
        args.seed = random.randint(0, 1000)
    print(f"Random seed is set to {args.seed}")
    fix_seed(args.seed, deterministic=args.reproduction)
    
    # Initialize the deep learning program
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    # Read event data
    path_save = path_result + 'EventEvaluation/'
    event_categories = sorted(os.listdir(path_result + 'EventData/'))
    event_data = pd.concat([pd.read_hdf(path_result + f'EventData/{event_cat}/event_data.h5', key='data') for event_cat in event_categories]).reset_index()
    if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
        event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    else:
        event_meta = pd.concat([pd.read_csv(path_result + f'EventData/{event_cat}/event_meta.csv') for event_cat in event_categories], ignore_index=True).set_index('event_id')
    danger_start = np.maximum(event_meta['impact_timestamp'].values-5000, event_meta['start_timestamp'].values)
    danger_end = np.minimum(event_meta['impact_timestamp'].values+500, event_meta['end_timestamp'].values)
    event_meta['danger_start'] = danger_start
    event_meta['danger_end'] = danger_end
    assert np.all(np.isin(event_data['event_id'].unique(), event_meta.index.values))

    # Make sure use the same events as other methods
    event_id_list = []
    for event_cat in event_categories:
        event_featurs = np.load(path_result + f'EventData/{event_cat}/event_features.npz')
        event_id_list.append(event_featurs['event_id'])
    event_id_list = np.concatenate(event_id_list)
    event_id_list = pd.DataFrame(event_id_list, columns=['event_id','target_id','time'])
    event_data = event_data.merge(event_id_list, on=['event_id','target_id','time'], how='inner')
        
    event_data['vx_ego'] = event_data['v_ego']*event_data['hx_ego']
    event_data['vy_ego'] = event_data['v_ego']*event_data['hy_ego']
    event_data['vx_sur'] = event_data['v_sur']*event_data['hx_sur']
    event_data['vy_sur'] = event_data['v_sur']*event_data['hy_sur']
    veh_dimensions = event_meta[['ego_width','ego_length','target_width','target_length']].copy()
    condition = event_meta[['target_width','target_length']].isna().any(axis=1)
    veh_dimensions.loc[condition, ['target_width','target_length']] = event_meta.loc[condition, ['other_width','other_length']].values
    avg_width = np.nanmean(veh_dimensions['ego_width'].values)
    avg_length = np.nanmean(veh_dimensions['ego_length'].values)
    for var in ['ego_width','ego_length','target_width','target_length']:
        veh_dimensions.loc[veh_dimensions[var].isna(), var] = avg_width if 'width' in var else avg_length
    event_data[['length_ego','length_sur']] = veh_dimensions.loc[event_data['event_id'].values, ['ego_length','target_length']].values

    if os.path.exists(path_save + 'highD_ucd.h5'):
        safety_evaluation = pd.read_hdf(path_save + 'highD_ucd.h5', key='data')
    else:
        safety_evaluation = UCD(event_data, device)
        safety_evaluation.to_hdf(path_save + f'highD_ucd.h5', key='data', mode='w')

    ucd_thresholds = np.round(np.arange(1,100)**1.5)
    print('--- Analyzing ---')
    progress_bar = tqdm(ucd_thresholds, desc='UCD', ascii=True, dynamic_ncols=False, miniters=10)
    ucd_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'SSSE') for threshold in progress_bar)
    ucd_records = pd.concat(ucd_records).reset_index()
    ucd_records['indicator'] = 'UCD'
    ucd_records['model'] = 'ucd'

    ucd_records.loc[ucd_records['danger_recorded'].isna(), 'danger_recorded'] = False
    ucd_records.loc[ucd_records['safety_recorded'].isna(), 'safety_recorded'] = False
    ucd_records[['danger_recorded', 'safety_recorded']] = ucd_records[['danger_recorded', 'safety_recorded']].astype(bool)
    ucd_records[['indicator', 'model']] = ucd_records[['indicator', 'model']].astype(str)
    ucd_records.to_hdf(path_result + 'Analyses/Warning_ucd.h5', key='results', mode='w')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    manual_seed = 131
    path_result = './ResultData/'
    
    main(args, manual_seed, path_result)