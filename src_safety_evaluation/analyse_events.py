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

    '''
    Analysis 1 - Conflict detection comparison
    For each event, it is applicable to compare conflict detection if safety or danger period is present
    - danger: the period when an (near)crash happens as manually annotated by SHRP2
              * start: at most 4.5 seconds before impact_timestamp and after start_timestamp
              * end: 0.5 second after impact_timestamp and before end_timestamp
    - safety: the beginning in an event before start_timestamp with no hard braking
              * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
              * start: first evaluatable timestamp in the event
              * end: 0.5~5 seconds after the first timestamp, at least 3 seconds before start_timestamp
    The target has largest intensity/DRAC/EI (or smallest TTC/MTTC/PSD/TTC2D/TAdv/ACT) during danger period is considered 
    as the conflicting target, and the safe period is determined specifically
    Then the comparison of ROC curves is between different indicators under various thresholds
    '''
    danger_start = np.maximum(event_meta['impact_timestamp'].values-4500, event_meta['start_timestamp'].values)
    danger_end = np.minimum(event_meta['impact_timestamp'].values+500, event_meta['end_timestamp'].values)
    event_meta['danger_start'] = danger_start
    event_meta['danger_end'] = danger_end
    event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')
    
    # 1D and 2D SSMs
    for indicator in ['TTC', 'DRAC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT', 'EI']:
        if indicator in ['TTC', 'MTTC', 'TTC2D']:
            thresholds = np.unique(np.round(10**np.arange(0,1.68,0.015),1))-1
        elif indicator == 'DRAC':
            thresholds = np.unique(np.round(10**np.arange(0,1.,0.01),2))-1
        elif indicator == 'PSD':
            thresholds = np.unique(np.round((10**np.arange(0,2.25,0.015)-1)/50,2))
        elif indicator == 'TAdv':
            thresholds = np.unique(np.round((10**np.arange(0,2.1,0.02)-1)/10,2))
        elif indicator == 'ACT':
            thresholds = np.unique(np.round(10**np.arange(0,1.91,0.018),1))-1
        elif indicator == 'EI':
            thresholds = np.unique(np.round((10**np.arange(0,1.86,0.009)-1)/50,2))
        
        if os.path.exists(path_result + f'Analyses/Warning_{indicator}.h5'):
            print(f'--- Analysis 1 with {indicator} already completed ---')
        else:
            print(f'--- Analyzing with {indicator} ---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation(indicator, path_results)
            records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta, indicator) for threshold in thresholds)
            records = pd.concat(records).reset_index()
            records['indicator'] = indicator
            records['model'] = indicator
            records = fill_na_warning(records)
            records.to_hdf(path_result + f'Analyses/Warning_{indicator}.h5', key='results', mode='w')
            print(f'{indicator} time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    # SSSE
    model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
    dataset_name_list = model_evaluation['dataset'].values
    encoder_name_list = model_evaluation['encoder_selection'].values
    cross_attention_name_list = model_evaluation['cross_attention'].values
    pretraining_list = model_evaluation['pretraining'].values

    ssse_thresholds = np.unique(np.round(10**np.arange(0,4.2,0.035))).astype(int)
    for dataset_name, encoder_name, cross_attention_name, pretraining in zip(dataset_name_list, encoder_name_list, cross_attention_name_list, pretraining_list):
        model_name = f'{dataset_name}_{encoder_name}_{cross_attention_name}_{pretraining}'
        if os.path.exists(path_result + f'Analyses/Warning_{model_name}.h5'):
            print('--- Analysis 1 with', model_name, 'already completed ---')
        else:
            print('--- Analyzing with', model_name, '---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation('SSSE', path_results, dataset_name, encoder_name, cross_attention_name, pretraining)
            ssse_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta, 'SSSE') for threshold in ssse_thresholds)
            ssse_records = pd.concat(ssse_records).reset_index()
            ssse_records['indicator'] = 'SSSE'
            ssse_records['model'] = model_name
            ssse_records = fill_na_warning(ssse_records)
            ssse_records.to_hdf(path_result + f'Analyses/Warning_{model_name}.h5', key='results', mode='w')
            print(model_name, 'time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    print('--- Analysis 1: Conflict detection comparison completed ---')

    '''
    Analysis 2 - Warning timeliness
    Using the optimal threshold for every model,
    - optimal threshold: the threshold that makes true positive rate and false positive rate closest to (100%, 0%)
    for each event, the target has largest intensity/DRAC/EI (or smallest TTC/MTTC/PSD/TAdv/TTC2D/ACT) during danger period is considered 
    as the conflicting target; if a driver reaction is recorded, check if the first warning is before the reaction_timestamp
    - first warning: the last safe->unsafe transition moment before impact_timestamp
    '''
    if os.path.exists(path_result + 'Analyses/WarningTimeliness.h5'):
        print('--- Analysis 2: Part of warning timeliness already completed ---')
        existing_results = pd.read_hdf(path_result + 'Analyses/WarningTimeliness.h5', key='results')
        existing_models = existing_results['model'].unique()
    else:
        existing_models = []
        if os.path.exists(path_result + 'Analyses/EventMeta.csv'):
            event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
        if 'danger_start' not in event_meta.columns:
            danger_start = np.maximum(event_meta['impact_timestamp'].values-4500, event_meta['start_timestamp'].values)
            danger_end = np.minimum(event_meta['impact_timestamp'].values+500, event_meta['end_timestamp'].values)
            event_meta['danger_start'] = danger_start
            event_meta['danger_end'] = danger_end

        if 'conflict' not in event_meta.columns:
            event_meta = event_meta.set_index('event_id')
            for event_id in tqdm(event_meta.index.values):
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
            event_meta = event_meta.reset_index()

        results = []
        for conflict_indicator in ['TTC', 'DRAC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT', 'EI', 'UCD']:
            if conflict_indicator in existing_models:
                print('--- Warning timeliness with', conflict_indicator, 'already completed ---')
            else:
                print('--- Issuing warning', conflict_indicator, '---')
                conflict_warning = pd.read_hdf(path_result + f'Analyses/Warning_{conflict_indicator}.h5', key='results')
                safety_evaluation = read_evaluation(conflict_indicator, path_results)
                optimal_threshold = optimize_threshold(conflict_warning, conflict_indicator, 'ROC')
                if conflict_indicator == 'UCD':
                    records = issue_warning('SSSE', optimal_threshold, safety_evaluation, event_meta)
                else:
                    records = issue_warning(conflict_indicator, optimal_threshold, safety_evaluation, event_meta)
                records['model'] = conflict_indicator
                results.append(records.copy())

        for dataset_name, encoder_name, cross_attention_name, pretraining in zip(dataset_name_list, encoder_name_list, cross_attention_name_list, pretraining_list):
            model_name = f'{dataset_name}_{encoder_name}_{cross_attention_name}_{pretraining}'
            if model_name in existing_models:
                print('--- Warning timeliness with', model_name, 'already completed ---')
            else:
                print('--- Issuing warning', model_name, '---')
                conflict_warning = pd.read_hdf(path_result + f'Analyses/Warning_{model_name}.h5', key='results')
                safety_evaluation = read_evaluation('SSSE', path_results, dataset_name, encoder_name, cross_attention_name, pretraining)
                optimal_threshold = optimize_threshold(conflict_warning, 'SSSE', 'ROC')
                records = issue_warning('SSSE', optimal_threshold, safety_evaluation, event_meta)
                records['model'] = model_name
                results.append(records.copy())
        if len(results) > 0:
            results = pd.concat(results).reset_index()
            results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
            results['danger_recorded'] = results['danger_recorded'].astype(bool)
        if len(existing_models) > 0:
            results = pd.concat([results, existing_results]).reset_index(drop=True)
        results.to_hdf(path_result + 'Analyses/WarningTimeliness.h5', key='results', mode='w')
        event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')
        print('--- Analysis 2: Warning timeliness completed ---')
        print('Analysed models:', results['model'].unique())

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
