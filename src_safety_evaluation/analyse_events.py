'''
This script is to analyse the safety evaluation results for all events.
'''

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as systime
from joblib import Parallel, delayed
from validation_utils.utils_features import *
from src_safety_evaluation.validation_utils.utils_evaluation import *


def fill_na_warning(results):
    results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
    results.loc[results['safety_recorded'].isna(), 'safety_recorded'] = False
    results[['danger_recorded', 'safety_recorded']] = results[['danger_recorded', 'safety_recorded']].astype(bool)
    results[['indicator', 'model']] = results[['indicator', 'model']].astype(str)
    return results


def main(path_result):
    initial_time = systime.time()
    print('---- available cpus:', os.cpu_count(), '----')

    # Read data for all event categories
    path_events = path_result + 'EventData/'
    path_results = path_result + 'EventEvaluation/'
    event_meta, event_data = read_events(path_events)

    model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
    dataset_name_list = model_evaluation['dataset'].values
    encoder_name_list = model_evaluation['encoder_selection'].values
    cross_attention_name_list = model_evaluation['cross_attention'].values
    pretraining_list = model_evaluation['pretraining'].values

    # '''
    # Analysis 1 - Event severity
    # For each event, there could be 1 conflict or 2 sequential conflicts, of which
    # - danger_present: at least one target with long enough duration presents during [impact_timestamp-0.5 s, impact_timestamp+0.5 s]
    # - ground truth: Crash(3)/NearCrash(2)/CrashRelevant(1)/Not applicable(0)
    # - safety evaluation: average intensity in the period [impact_timestamp-0.5 s, impact_timestamp+0.5 s]
    # '''
    # if os.path.exists(path_result + 'Analyses/EventSeverity.h5'):
    #     print('--- Analysis 1: Event severity completed ---')
    # else:
    #     ground_truth = event_meta[event_meta['duration_enough']]
    #     results = []
    #     for dataset_name, encoder_name, cross_attention_name, pretraining in zip(dataset_name_list, encoder_name_list, cross_attention_name_list, pretraining_list):
    #         safety_evaluation = read_evaluation('SSSE', path_results, dataset_name, encoder_name, cross_attention_name, pretraining)
    #         safety_evaluation = safety_evaluation[safety_evaluation['event_id'].isin(ground_truth.index.values)]
    #         safety_evaluation['impact_time'] = ground_truth.loc[safety_evaluation['event_id'].values, 'impact_timestamp'].values/1000
    #         safety_evaluation['period_start'] = safety_evaluation['impact_time'] - 0.5
    #         safety_evaluation['period_end'] = safety_evaluation['impact_time'] + 0.5
    #         safety_evaluation = safety_evaluation[(safety_evaluation['time']>=safety_evaluation['period_start'])&
    #                                             (safety_evaluation['time']<=safety_evaluation['period_end'])]
    #         avg_intensity = safety_evaluation.groupby(['event_id','target_id'])['intensity'].mean().reset_index().set_index('event_id')
    #         result = []
    #         for event_id in tqdm(avg_intensity.index.unique(), desc=pretraining+'_'+encoder_name+'_'+cross_attention_name, ascii=True, dynamic_ncols=False):
    #             if ground_truth.loc[event_id, 'severity_first'] >= ground_truth.loc[event_id, 'severity_second']:
    #                 severity_higher = ground_truth.loc[event_id, 'severity_first']
    #                 severity_lower = ground_truth.loc[event_id, 'severity_second']
    #             else:
    #                 severity_higher = ground_truth.loc[event_id, 'severity_second']
    #                 severity_lower = ground_truth.loc[event_id, 'severity_first']
    #             if severity_higher>0 or severity_lower>0:
    #                 event_meta.loc[event_id, 'danger_present'] = True
    #             sorted_intensity = avg_intensity.loc[[event_id]].sort_values('intensity', ascending=False)
    #             intensity_higher = sorted_intensity.iloc[0]['intensity']
    #             intensity_lower = sorted_intensity.iloc[1]['intensity'] if len(sorted_intensity)>1 else 0.
    #             result.append([str(event_id)+'-higher', severity_higher, intensity_higher])
    #             result.append([str(event_id)+'-lower', severity_lower, intensity_lower])
    #         result = pd.DataFrame(result, columns=['event','given_severity','evaluated_intensity'])
    #         result['model'] = pretraining + '_' + encoder_name + '_' + cross_attention_name
    #         results.append(result)
    #     results = pd.concat(results)
    #     results.to_hdf(path_result + f'Analyses/EventSeverity.h5', key='results', mode='w')
    #     event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')
    #     print('--- Analysis 1: Event severity completed ---')

    '''
    Analysis 2 - Conflict detection comparison
    For each event, it is applicable to compare conflict detection if safety or danger period is present
    - danger: the period when an (near)crash happens as manually annotated by SHRP2
              * start: at most 5 seconds before impact_timestamp and after start_timestamp
              * end: 0.5 second after impact_timestamp and before end_timestamp
    - safety: the beginning in an event before start_timestamp with no hard braking
              * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
              * start: first evaluatable timestamp in the event
              * end: 0.5~5 seconds after the first timestamp, at least 3 seconds before start_timestamp
    The target has largest intensity/DRAC (or smallest TTC/MTTC) during danger period is considered 
    as the conflicting target, and the safe period is determined specifically
    Then the comparison of ROC curves is between different indicators under various thresholds
    '''
    if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
        event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    danger_start = np.maximum(event_meta['impact_timestamp'].values-5000, event_meta['start_timestamp'].values)
    danger_end = np.minimum(event_meta['impact_timestamp'].values+500, event_meta['end_timestamp'].values)
    event_meta['danger_start'] = danger_start
    event_meta['danger_end'] = danger_end
    event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')
    
    if os.path.exists(path_result + 'Analyses/Warning_ttc.h5'):
        print('--- Analysis 2 with TTC already completed ---')
    else:
        print('--- Analyzing with TTC ---')
        sub_initial_time = systime.time()
        ttc_thresholds = np.round(np.arange(0.2,20.,0.2)**1.2,1)
        safety_evaluation = read_evaluation('TTC', path_results)
        ttc_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'TTC') for threshold in ttc_thresholds)
        ttc_records = pd.concat(ttc_records).reset_index()
        ttc_records['indicator'] = 'TTC'
        ttc_records['model'] = 'ttc'
        ttc_records = fill_na_warning(ttc_records)
        ttc_records.to_hdf(path_result + 'Analyses/Warning_ttc.h5', key='results', mode='w')
        print('TTC time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    if os.path.exists(path_result + 'Analyses/Warning_drac.h5'):
        print('--- Analysis 2 with DRAC already completed ---')
    else:
        print('--- Analyzing with DRAC ---')
        sub_initial_time = systime.time()
        drac_thresholds = np.round(np.arange(0.05,5.,0.05)**1.4, 2)
        safety_evaluation = read_evaluation('DRAC', path_results)
        drac_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'DRAC') for threshold in drac_thresholds)
        drac_records = pd.concat(drac_records).reset_index()
        drac_records['indicator'] = 'DRAC'
        drac_records['model'] = 'drac'
        drac_records = fill_na_warning(drac_records)
        drac_records.to_hdf(path_result + 'Analyses/Warning_drac.h5', key='results', mode='w')
        print('DRAC time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    if os.path.exists(path_result + 'Analyses/Warning_mttc.h5'):
        print('--- Analysis 2 with MTTC already completed ---')
    else:
        print('--- Analyzing with MTTC ---')
        sub_initial_time = systime.time()
        mttc_thresholds = np.round(np.arange(0.2,20.,0.2),1)
        safety_evaluation = read_evaluation('MTTC', path_results)
        mttc_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'MTTC') for threshold in mttc_thresholds)
        mttc_records = pd.concat(mttc_records).reset_index()
        mttc_records['indicator'] = 'MTTC'
        mttc_records['model'] = 'mttc'
        mttc_records = fill_na_warning(mttc_records)
        mttc_records.to_hdf(path_result + 'Analyses/Warning_mttc.h5', key='results', mode='w')
        print('MTTC time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    ssse_thresholds = np.round(np.arange(1,100)**1.5)
    for dataset_name, encoder_name, cross_attention_name, pretraining in zip(dataset_name_list, encoder_name_list, cross_attention_name_list, pretraining_list):
        model_name = f'{dataset_name}_{encoder_name}_{cross_attention_name}_{pretraining}'
        if os.path.exists(path_result + f'Analyses/Warning_{model_name}.h5'):
            print('--- Analysis 2 with', model_name, 'already completed ---')
        else:
            print('--- Analyzing with', model_name, '---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation('SSSE', path_results, dataset_name, encoder_name, cross_attention_name, pretraining)
            ssse_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'SSSE') for threshold in ssse_thresholds)
            ssse_records = pd.concat(ssse_records).reset_index()
            ssse_records['indicator'] = 'SSSE'
            ssse_records['model'] = model_name
            ssse_records = fill_na_warning(ssse_records)
            ssse_records.to_hdf(path_result + f'Analyses/Warning_{model_name}.h5', key='results', mode='w')
            print(model_name, 'time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    print('--- Analysis 2: Conflict detection comparison completed ---')

    # '''
    # Analysis 3 - Warning timeliness
    # Using the optimal threshold for every model,
    # - optimal threshold: the threshold that makes true positive rate and false positive rate closest to (100%, 0%)
    # for each event, the target has largest intensity/DRAC (or smallest TTC/MTTC) during danger period is considered 
    # as the conflicting target; if a driver reaction is recorded, check if the first warning is before the reaction_timestamp
    # - first warning: the last safe->unsafe transition moment before impact_timestamp
    # '''
    # if os.path.exists(path_result + 'Analyses/WarningTimeliness.h5'):
    #     print('--- Analysis 3: Warning timeliness completed ---')
    # else:
    #     if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
    #         event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    #     if 'danger_start' not in event_meta.columns:
    #         danger_start = np.maximum(event_meta['impact_timestamp'].values-5000, event_meta['start_timestamp'].values)
    #         danger_end = np.minimum(event_meta['impact_timestamp'].values+500, event_meta['end_timestamp'].values)
    #         event_meta['danger_start'] = danger_start
    #         event_meta['danger_end'] = danger_end
        
    #     results = []
    #     for conflict_indicator, model in zip(['DRAC', 'TTC', 'MTTC'], ['drac', 'ttc', 'mttc']):
    #         conflict_warning = pd.read_hdf(path_result + f'Analyses/Warning_{model}.h5', key='results')
    #         filtered_warning = conflict_warning[(~conflict_warning['false warning'].isna())&
    #                                             (~conflict_warning['true warning'].isna())&
    #                                             (conflict_warning['model']==model)]
    #         safety_evaluation = read_evaluation(conflict_indicator, path_results)
    #         optimal_threshold = optimize_threshold(filtered_warning, conflict_indicator)
    #         records = issue_warning(conflict_indicator, optimal_threshold, safety_evaluation, event_data, event_meta)
    #         records['model'] = model
    #         results.append(records)

    #     for pretraining, encoder_name, cross_attention_name in zip(pretraining_list, encoder_name_list, cross_attention_name_list):
    #         model_name = pretraining + '_' + encoder_name + '_' + cross_attention_name
    #         filtered_warning = conflict_warning[(~conflict_warning['false warning'].isna())&
    #                                             (~conflict_warning['true warning'].isna())&
    #                                             (conflict_warning['model']==model_name)]
    #         print('--- Analyzing', model_name, '---')
    #         safety_evaluation = read_evaluation('SSSE', path_results, pretraining, encoder_name, cross_attention_name)
    #         optimal_threshold = optimize_threshold(filtered_warning, 'SSSE')
    #         records = issue_warning('SSSE', optimal_threshold, safety_evaluation, event_data, event_meta)
    #         records['model'] = model_name
    #         results.append(records)

    #     results = pd.concat(results).reset_index()
    #     results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
    #     results['danger_recorded'] = results['danger_recorded']].astype(bool)
    #     results.to_hdf(path_result + 'Analyses/WarningTimeliness.h5', key='results', mode='w')
    #     event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')
    #     print('--- Analysis 3: Warning timeliness completed ---')

    # '''
    # Save the identified target by different models under corresponding optimal thresholds    
    # '''
    # if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
    #     event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    # warning_timeliness = pd.read_hdf(path_result + 'Analyses/WarningTimeliness.h5', key='results')
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

    path_result = './ResultData/'
    os.makedirs(path_result + 'Analyses/', exist_ok=True)

    main(path_result)
