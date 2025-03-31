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
from src_safety_evaluation.validation_utils.utils_evaluation import read_evaluation, optimize_threshold, issue_warning, parallel_records, read_events


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


def get_model_fig(model_name):
    dataset_name = model_name.split('_current')[0]

    features = model_name.split('pretrained')[0]
    if 'acc' in features:
        features = features.split('current+acc_')[1]
        features = features.split('_')
        features = ['current+acc'] + features
    else:
        features = features.split('current')[1]
        features = features.split('_')
        features = ['current'] + features
    features = [f for f in features if f not in ['not','']]
    encoder_name = '_'.join(features)

    if 'not_pretrained' in model_name:
        pretrained = 'not_pretrained'
    elif 'pretrained_all' in model_name:
        pretrained = 'pretrained_all'
    else:
        pretrained = 'pretrained'
        
    return dataset_name, encoder_name, pretrained


def main(args, path_result, path_prepared):
    initial_time = systime.time()
    print('---- available cpus:', os.cpu_count(), '----')

    # Read data for all event categories
    path_events = path_result + 'EventData/'
    path_results = path_result + 'EventEvaluation/'
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
    Analysis 1 - Conflict detection
    For each event, it is applicable to compare conflict detection if safety or danger period is present
    - danger: the period when an (near)crash happens as manually annotated by SHRP2
              * start: after start_timestamp or within 4.5 seconds before impact_timestamp
              * end: 0.5 second after impact_timestamp and before end_timestamp
    - safety: the beginning in an event before start_timestamp with no hard braking
              * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
              * start: first evaluatable timestamp in the event
              * end: 0.5~5 seconds after the first timestamp, at least 3 seconds before start_timestamp
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
            safety_evaluation = read_evaluation(indicator, path_results)
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
    evaluation_files = os.listdir(path_results)
    evaluation_files = [f for f in evaluation_files if f.endswith('.h5') and f!='TAdv_TTC2D_ACT_EI.h5' and f!='highD_UCD.h5']
    if args.reversed_list:
        evaluation_files = evaluation_files[::-1]

    gssm_thresholds = np.unique(np.round(np.arange(0,6,0.06)-0.06,2))
    for model_name in evaluation_files:
        if os.path.exists(path_result + f'Analyses/Warning_{model_name}.h5'):
            print('--- Analysis 1 with', model_name, 'already completed ---')
        else:
            print('--- Analyzing with', model_name, '---')
            dataset_name, encoder_name, pretraining = get_model_fig(model_name)
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation('GSSM', path_results, dataset_name, encoder_name, pretraining)
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

    '''
    Analysis 2 - Warning timeliness and warning period
    For every model, the optimal threshold makes false negative rate and false positive rate be closest to (0%, 0%).
    For each event, the target has largest intensity/EI (or smallest TAdv/TTC2D/ACT) during danger
    period is considered as the conflicting target, then
    - first warning: the last safe->unsafe transition moment before impact_timestamp,
    - warning period: the percentage of warned time moments within [danger_start, danger_end].
    '''
    if os.path.exists(path_result + 'Analyses/OptimalWarningEvaluation.h5'):
        print('--- Analysis 2: Part of optimal warning analysis already completed ---')
        existing_results = pd.read_hdf(path_result + 'Analyses/OptimalWarningEvaluation.h5', key='results')
        existing_models = existing_results['model'].unique()
    else:
        existing_models = []

    results = []
    for conflict_indicator in ['TAdv', 'TTC2D', 'ACT', 'EI', 'UCD']:
        if conflict_indicator in existing_models:
            print('--- Optimal warning analysis with', conflict_indicator, 'already completed ---')
        else:
            print('--- Issuing warning', conflict_indicator, '---')
            conflict_warning = pd.read_hdf(path_result + f'Analyses/Warning_{conflict_indicator}.h5', key='results')
            safety_evaluation = read_evaluation(conflict_indicator, path_results)
            if conflict_indicator == 'UCD':
                optimal_threshold = optimize_threshold(conflict_warning, 'GSSM', 'ROC')
                records = issue_warning('GSSM', optimal_threshold, safety_evaluation, event_meta)
            else:
                optimal_threshold = optimize_threshold(conflict_warning, conflict_indicator, 'ROC')
                records = issue_warning(conflict_indicator, optimal_threshold, safety_evaluation, event_meta)
            records['model'] = conflict_indicator
            results.append(records.copy())

    for dataset_name, encoder_name, pretraining in zip(dataset_name_list, encoder_name_list, pretraining_list):
        model_name = f'{dataset_name}_{encoder_name}_{pretraining}'
        if model_name in existing_models:
            print('--- Optimal warning analysis with', model_name, 'already completed ---')
        else:
            print('--- Issuing warning', model_name, '---')
            conflict_warning = pd.read_hdf(path_result + f'Analyses/Warning_{model_name}.h5', key='results')
            safety_evaluation = read_evaluation('GSSM', path_results, dataset_name, encoder_name, pretraining)
            optimal_threshold = optimize_threshold(conflict_warning, 'GSSM', 'ROC')
            records = issue_warning('GSSM', optimal_threshold, safety_evaluation, event_meta)
            records['model'] = model_name
            results.append(records.copy())
    if len(results) > 0:
        results = pd.concat(results).reset_index()
        results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
        results['danger_recorded'] = results['danger_recorded'].astype(bool)
    if len(existing_models) > 0:
        results = pd.concat([results, existing_results]).reset_index(drop=True)
    results.to_hdf(path_result + 'Analyses/OptimalWarningEvaluation.h5', key='results', mode='w')
    print('--- Optimal warning analysis completed ---')
    print('Analysed models:', results['model'].unique())

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    path_result = 'ResultData/'
    os.makedirs(path_result + 'Analyses/', exist_ok=True)
    path_prepared = 'PreparedData/'

    args = parse_args()
    main(args, path_result, path_prepared)
