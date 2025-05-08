'''
This script is to analyse the safety evaluation results for all events.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as systime
from joblib import Parallel, delayed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_safety_evaluation.validation_utils.utils_evaluation import read_evaluation, parallel_records, read_events
manual_seed = 131


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reversed_list', type=int, default=0, help='Whether to reverse the model list (defaults to False), useful for running parallel jobs')
    args = parser.parse_args()
    args.reversed_list = bool(args.reversed_list)
    return args


def fill_na_warning(results):
    results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
    results.loc[results['safety_recorded'].isna(), 'safety_recorded'] = False
    results[['danger_recorded', 'safety_recorded']] = results[['danger_recorded', 'safety_recorded']].astype(bool)
    results[['safe_target_ids', 'indicator', 'model']] = results[['safe_target_ids', 'indicator', 'model']].astype(str)
    return results


def main(args, path_result):
    initial_time = systime.time()
    print('---- available cpus:', os.cpu_count(), '----')

    # Read data for all event categories
    path_events = path_result + 'EventData/'
    path_eval = path_result + 'EventEvaluation/'
    event_meta, event_data = read_events(path_events)
    if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
        event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    if 'danger_start' not in event_meta.columns:
        danger_start = np.minimum(event_meta['impact_timestamp'].values-4500, event_meta['start_timestamp'].values)
        danger_end = np.minimum(event_meta['impact_timestamp'].values+500, event_meta['end_timestamp'].values)
        event_meta['danger_start'] = danger_start
        event_meta['danger_end'] = danger_end
        event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')

    '''
    For each event, it is applicable to compare conflict detection if safety or danger period is present
    - danger: the period when an (near)crash happens as manually annotated by SHRP2
              * start: after start_timestamp or within 4.5 seconds before impact_timestamp
              * end: 0.5 second after impact_timestamp and before end_timestamp
    - safety: for each object other than the conflicting target, before impact_timestamp with no hard braking
              * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
              * start: 1.5 seconds after an object is detected
              * end: 2~5 seconds after the first timestamp, at least 3 seconds before start_timestamp
    The target has largest intensity/EI (or smallest TAdv/TTC2D/ACT) during danger period is 
    considered as the conflicting target, and the safe period is determined for other vehicles than the target.
    Then conflct detection are implemented with different indicators under various thresholds.
    '''
    
    # 2D SSMs
    for indicator in ['TAdv', 'TTC2D', 'ACT', 'EI']:
        if indicator == 'TAdv':
            thresholds = np.unique(np.round(np.arange(0,1.75,0.0115)**7,2))
        elif indicator in ['TTC2D', 'ACT']:
            thresholds = np.unique(np.round(np.arange(0,1.94,0.0135)**7,2))
        elif indicator == 'EI':
            thresholds = np.round((8**np.arange(0,2.31,0.0265)-1)/50, 2)
            thresholds = np.unique(np.sort(np.concatenate([thresholds, -thresholds[::2]*3])))
        
        if os.path.exists(path_result + f'Analyses/Warning_{indicator}.h5'):
            print(f'--- Analysis 1 with {indicator} already completed ---')
        else:
            print(f'--- Analyzing with {indicator} ---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation(indicator, path_eval)
            progress_bar = tqdm(thresholds, desc=indicator, ascii=True, dynamic_ncols=False, miniters=10)
            records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta, indicator) for threshold in progress_bar)
            records = pd.concat(records).reset_index()
            records['indicator'] = indicator
            records['model'] = indicator
            records = fill_na_warning(records)
            records.to_hdf(path_result + f'Analyses/Warning_{indicator}.h5', key='results', mode='w')
            progress_bar.close()
            print(f'{indicator} time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    # GSSM
    evaluation_files = sorted(os.listdir(path_eval), key=lambda f: os.path.getmtime(os.path.join(path_eval, f)))
    evaluation_files = [f[:-3] for f in evaluation_files if f.endswith('.h5') and f!='TAdv_TTC2D_ACT_EI.h5' and 'UCD' not in f]
    if args.reversed_list:
        evaluation_files = evaluation_files[::-1]

    gssm_thresholds = np.unique(np.round(np.arange(0,6,0.06)-0.06,2))
    for model_name in evaluation_files:
        if os.path.exists(path_result + f'Analyses/Warning_{model_name}.h5'):
            print('--- Analysis 1 with', model_name, 'already completed ---')
        else:
            print('--- Analyzing with', model_name, '---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation('GSSM', path_eval, model_name)
            progress_bar = tqdm(gssm_thresholds, desc=model_name, ascii=True, dynamic_ncols=False, miniters=10)
            gssm_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta, 'GSSM') for threshold in progress_bar)
            gssm_records = pd.concat(gssm_records).reset_index()
            gssm_records['indicator'] = 'GSSM'
            gssm_records['model'] = model_name
            gssm_records = fill_na_warning(gssm_records)
            gssm_records.to_hdf(path_result + f'Analyses/Warning_{model_name}.h5', key='results', mode='w')
            progress_bar.close()
            print(model_name, 'time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    print('--- Conflict detection comparison completed ---')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(manual_seed)
    path_result = 'ResultData/'
    os.makedirs(path_result + 'Analyses/', exist_ok=True)
    args = parse_args()
    main(args, path_result)
