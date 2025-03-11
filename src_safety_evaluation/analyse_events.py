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
    Analysis 1 - Conflict detection comparison
    For each event, it is applicable to compare conflict detection if safety or danger period is present
    - danger: the period when an (near)crash happens as manually annotated by SHRP2
              * start: after start_timestamp or within 4.5 seconds before impact_timestamp
              * end: 0.5 second after impact_timestamp and before end_timestamp
    - safety: the beginning in an event before start_timestamp with no hard braking
              * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
              * start: first evaluatable timestamp in the event
              * end: 0.5~5 seconds after the first timestamp, at least 3 seconds before start_timestamp
    The target has largest intensity/DRAC/EI (or smallest TTC/MTTC/PSD/TTC2D/TAdv/ACT) during danger period is considered 
    as the conflicting target, and the safe period is determined specifically
    Then the comparison of ROC curves is between different indicators under various thresholds
    '''
    
    # 1D and 2D SSMs
    for indicator in ['TTC', 'DRAC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT', 'EI']:
        if indicator in ['TTC', 'MTTC', 'TTC2D']:
            thresholds = np.round(np.unique(np.round(10**np.arange(0,1.68,0.015),1))-0.9, 1)
        elif indicator == 'DRAC':
            thresholds = np.round(np.unique(np.round(10**np.arange(0,1.,0.01),2))-1, 2)
        elif indicator == 'PSD':
            thresholds = np.unique(np.round((10**np.arange(0,2.25,0.015)-1)/50, 2))
        elif indicator == 'TAdv':
            thresholds = np.unique(np.round((10**np.arange(0,2.1,0.02)-0.9)/10, 2))
        elif indicator == 'ACT':
            thresholds = np.round(np.unique(np.round(10**np.arange(0,1.91,0.018),1))-0.9, 1)
        elif indicator == 'EI':
            thresholds = np.unique(np.round((10**np.arange(0,1.86,0.009)-0.5)/50, 2))
        
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

    # SSSE
    model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
    dataset_name_list = model_evaluation['dataset'].values
    encoder_name_list = model_evaluation['encoder_selection'].values
    pretraining_list = model_evaluation['pretraining'].values
    if args.reversed_list:
        dataset_name_list = dataset_name_list[::-1]
        encoder_name_list = encoder_name_list[::-1]
        pretraining_list = pretraining_list[::-1]

    ssse_thresholds = np.unique(np.round(np.arange(0,20,0.2)**0.5,2))
    for dataset_name, encoder_name, pretraining in zip(dataset_name_list, encoder_name_list, pretraining_list):
        model_name = f'{dataset_name}_{encoder_name}_{pretraining}'
        if os.path.exists(path_result + f'Analyses/Warning_{model_name}.h5'):
            print('--- Analysis 1 with', model_name, 'already completed ---')
        else:
            print('--- Analyzing with', model_name, '---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation('SSSE', path_results, dataset_name, encoder_name, pretraining)
            progress_bar = tqdm(ssse_thresholds, desc=model_name, ascii=True, dynamic_ncols=False, miniters=10)
            ssse_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta, 'SSSE') for threshold in progress_bar)
            ssse_records = pd.concat(ssse_records).reset_index()
            ssse_records['indicator'] = 'SSSE'
            ssse_records['model'] = model_name
            ssse_records = fill_na_warning(ssse_records)
            ssse_records.to_hdf(path_result + f'Analyses/Warning_{model_name}.h5', key='results', mode='w')
            progress_bar.close()
            print(model_name, 'time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    print('--- Analysis 1: Conflict detection comparison completed ---')

    '''
    Analysis 2 - Warning timeliness and warning period
    Using the optimal threshold for every model,
    - optimal threshold: the threshold that makes true positive rate and false positive rate closest to (100%, 0%)
    for each event, the target has largest intensity/DRAC/EI (or smallest TTC/MTTC/PSD/TAdv/TTC2D/ACT) during danger period is considered 
    as the conflicting target; if a driver reaction is recorded, check if the first warning is before the reaction_timestamp
    - first warning: the last safe->unsafe transition moment before impact_timestamp
    - warning period: the percentage of warned time moments within [danger_start, danger_end]
    '''
    if os.path.exists(path_result + 'Analyses/OptimalWarningEvaluation.h5'):
        print('--- Analysis 2: Part of optimal warning analysis already completed ---')
        existing_results = pd.read_hdf(path_result + 'Analyses/OptimalWarningEvaluation.h5', key='results')
        existing_models = existing_results['model'].unique()
    else:
        existing_models = []

    if 'conflict' not in event_meta.columns:
        for event_id in tqdm(event_meta.index.values, desc='Severity', ascii=True, dynamic_ncols=False, miniters=714):
            for order in ['first', 'second']:
                '''
                This dataset does not record objects behind the ego vehicle;
                neither does it detect obstacles that have not shape.
                '''
                if event_meta.loc[event_id, order] in ['following', 'obstacle', 'single']:
                    event_meta.loc[event_id, f'severity_{order}'] = 0
            
            if event_meta.loc[event_id, 'severity_first'] < 0.5 and event_meta.loc[event_id, 'severity_second'] < 0.5:
                event_meta.loc[event_id, 'conflict'] = 'none'
            elif event_meta.loc[event_id, 'severity_first'] > event_meta.loc[event_id, 'severity_second']:
                event_meta.loc[event_id, 'conflict'] = event_meta.loc[event_id, 'first']
            elif event_meta.loc[event_id, 'severity_second'] > event_meta.loc[event_id, 'severity_first']:
                event_meta.loc[event_id, 'conflict'] = event_meta.loc[event_id, 'second']
            else:
                event_meta.loc[event_id, 'conflict'] = event_meta.loc[event_id, 'first']
    event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')

    # Filter out events of which the conflict was not detected
    event_meta = event_meta[event_meta['conflict']!='none']
    filtered_events = event_meta.index.values

    results = []
    for conflict_indicator in ['TTC', 'DRAC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT', 'EI', 'UCD']:
        if conflict_indicator in existing_models:
            print('--- Optimal warning analysis with', conflict_indicator, 'already completed ---')
        else:
            print('--- Issuing warning', conflict_indicator, '---')
            conflict_warning = pd.read_hdf(path_result + f'Analyses/Warning_{conflict_indicator}.h5', key='results')
            safety_evaluation = read_evaluation(conflict_indicator, path_results)
            filtered_warning = conflict_warning[conflict_warning['event_id'].isin(filtered_events)]
            if conflict_indicator == 'UCD':
                optimal_threshold = optimize_threshold(filtered_warning, 'SSSE', 'ROC')
                records = issue_warning('SSSE', optimal_threshold, safety_evaluation, event_meta)
            else:
                optimal_threshold = optimize_threshold(filtered_warning, conflict_indicator, 'ROC')
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
            safety_evaluation = read_evaluation('SSSE', path_results, dataset_name, encoder_name, pretraining)
            filtered_warning = conflict_warning[conflict_warning['event_id'].isin(filtered_events)]
            optimal_threshold = optimize_threshold(filtered_warning, 'SSSE', 'ROC')
            records = issue_warning('SSSE', optimal_threshold, safety_evaluation, event_meta)
            records['model'] = model_name
            results.append(records.copy())
    if len(results) > 0:
        results = pd.concat(results).reset_index()
        results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
        results['danger_recorded'] = results['danger_recorded'].astype(bool)
    if len(existing_models) > 0:
        results = pd.concat([results, existing_results]).reset_index(drop=True)
    results.to_hdf(path_result + 'Analyses/OptimalWarningEvaluation.h5', key='results', mode='w')
    print('--- Analysis 2: Optimal warning analysis completed ---')
    print('Analysed models:', results['model'].unique())

    # '''
    # Save the identified target by the best models in each category under corresponding optimal thresholds    
    # '''
    # if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
    #     event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    # warning_timeliness = pd.read_hdf(path_result + 'Analyses/OptimalWarningEvaluation.h5', key='results')
    # for model in warning_timeliness['model'].unique():
    #     warning_model = warning_timeliness[warning_timeliness['model']==model]
    #     event_meta.loc[warning_model['event_id'].values, 'target_id'] = warning_model['target_id'].values
    # event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')

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
