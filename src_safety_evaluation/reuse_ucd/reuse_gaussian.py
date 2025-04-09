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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_posterior_inference.model import LogNormalNLL
from src_safety_evaluation.validation_utils.utils_evaluation import parallel_records
from src_safety_evaluation.reuse_ucd.unified_conflit_detection import UCD, define_model
from joblib import Parallel, delayed


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

    path_save = path_result + 'EventEvaluation/'
    os.makedirs(path_save, exist_ok=True)
    os.makedirs(path_result + 'Analyses/', exist_ok=True)

    # Safety evaluation, make sure use the same events as other methods
    event_categories = sorted(os.listdir(path_result + 'EventData/'))
    current_features = []
    spacing_list = []
    event_id_list = []
    for event_cat in event_categories:
        event_featurs = np.load(path_result + f'EventData/{event_cat}/event_features.npz')
        current_features.append(event_featurs['current'])
        spacing_list.append(event_featurs['spacing'])
        event_id_list.append(event_featurs['event_id'])
    current_features = np.concatenate(current_features, axis=0) # [num_moments, 13]
    spacing_list = np.concatenate(spacing_list, axis=0) # [num_moments]
    event_id_list = np.concatenate(event_id_list, axis=0) # [num_moments, 3]
    '''
    features: ['length_ego','length_sur','combined_width',
               'vy_ego','vx_sur','vy_sur','v_ego2','v_sur2','delta_v2','delta_v',
               'psi_sur','rho']
    '''
    interaction_context = current_features[:,list(range(11))+[-1]]
    states = (interaction_context, spacing_list)

    # Load trained model
    model, likelihood = define_model(100, device)

    # Evaluation
    if os.path.exists(path_save + f'SafeBaseline_UCD.h5'):
        results = pd.read_hdf(path_save + f'SafeBaseline_UCD.h5', key='data')
        print(f'The events has been evaluated by UCD.')
    else:
        if os.path.exists(path_save + 'EvaluationEfficiency.csv'):
            eval_efficiency = pd.read_csv(path_save + 'EvaluationEfficiency.csv', dtype={'model_name':str,'time':float,'num_targets':int,'num_moments':int})
        else:
            eval_efficiency = pd.DataFrame(columns=['model_name','time','num_targets','num_moments'])
            eval_efficiency.to_csv(path_save + 'EvaluationEfficiency.csv', index=False)

        time_start = systime.time()
        mu, sigma, max_intensity = UCD(states, model, likelihood, device)
        time_end = systime.time()

        results = pd.DataFrame(event_id_list, columns=['event_id','target_id','time'])
        results[['event_id','target_id']] = results[['event_id','target_id']].astype(int)
        results['proximity'] = spacing_list
        results['mu'] = mu
        results['sigma'] = sigma
        results['intensity'] = max_intensity
        results = results.sort_values(['event_id','target_id','time']).reset_index(drop=True)
        results.to_hdf(path_save + f'SafeBaseline_UCD.h5', key='data', mode='w')
        eval_efficiency.loc[len(eval_efficiency)] = ['UCD', time_end-time_start, results['target_id'].nunique(), len(results)]
        eval_efficiency.to_csv(path_save + 'EvaluationEfficiency.csv', index=False)
        results['mode'] = np.exp(results['mu'] - results['sigma']**2)
        print(results[['mu','sigma','mode']].describe().to_string(float_format=lambda x: f'{x:.4f}'))
        loss_func = LogNormalNLL()
        log_var = np.log(np.maximum(1e-6, sigma**2))
        out = (torch.from_numpy(mu).float(), torch.from_numpy(log_var).float())
        print(f'LogNormal NLL: {loss_func(out, torch.from_numpy(spacing_list).float()).item()}')

    # Warning analysis
    event_data = pd.concat([pd.read_hdf(path_result + f'EventData/{event_cat}/event_data.h5', key='data') for event_cat in event_categories]).reset_index()
    if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
        event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    else:
        event_meta = pd.concat([pd.read_csv(path_result + f'EventData/{event_cat}/event_meta.csv') for event_cat in event_categories], ignore_index=True).set_index('event_id')
    danger_start = np.minimum(event_meta['impact_timestamp'].values-4500, event_meta['start_timestamp'].values)
    danger_end = np.minimum(event_meta['impact_timestamp'].values+500, event_meta['end_timestamp'].values)
    event_meta['danger_start'] = danger_start
    event_meta['danger_end'] = danger_end
    assert np.all(np.isin(event_data['event_id'].unique(), event_meta.index.values))

    ucd_thresholds = np.unique(np.round(10**np.arange(0,5.95,0.055))-2)
    print('--- Analyzing ---')
    progress_bar = tqdm(ucd_thresholds, desc='UCD', ascii=True, dynamic_ncols=False, miniters=10)
    ucd_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, results, event_data, event_meta, 'GSSM') for threshold in progress_bar)
    ucd_records = pd.concat(ucd_records).reset_index()
    ucd_records['indicator'] = 'UCD'
    ucd_records['model'] = 'UCD'
    progress_bar.close()

    ucd_records.loc[ucd_records['danger_recorded'].isna(), 'danger_recorded'] = False
    ucd_records.loc[ucd_records['safety_recorded'].isna(), 'safety_recorded'] = False
    ucd_records[['danger_recorded', 'safety_recorded']] = ucd_records[['danger_recorded', 'safety_recorded']].astype(bool)
    ucd_records[['safe_target_ids', 'indicator', 'model']] = ucd_records[['safe_target_ids', 'indicator', 'model']].astype(str)
    ucd_records.to_hdf(path_result + 'Analyses/Warning_UCD.h5', key='results', mode='w')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    manual_seed = 131
    path_result = 'ResultData/'
    
    main(args, manual_seed, path_result)