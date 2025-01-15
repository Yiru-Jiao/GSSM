'''
'''

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as systime
from joblib import Parallel, delayed
from validation_utils.utils_features import *
from validation_utils.utils_detection import *


def main(path_prepared, path_result):
    initial_time = systime.time()
    print('---- available cpus:', os.cpu_count(), '----')

    # Read data for all event categories
    path_events = path_result + 'EventEvaluation/'
    event_meta, event_data = read_events(path_events)

    pretraining_list = ['not_pretrained']*7 + ['pretrained']*7
    encoder_name_list = ['current', 'current_environment'] + ['current_environment_profiles']*5
    encoder_name_list = encoder_name_list*2
    cross_attention_name_list = ['not_crossed']*3 + ['first', 'last', 'first_last', 'first_middle_last']
    cross_attention_name_list = cross_attention_name_list*2

    '''
    Analysis 1 - Event severity
    For each event, there could be 1 conflict or 2 sequential conflicts, of which
    - danger_present: at least one target with long enough duration presents during [impact_timestamp-0.5 s, impact_timestamp+0.5 s]
    - ground truth: Crash(3)/NearCrash(2)/CrashRelevant(1)/Not applicable(0)
    - safety evaluation: average intensity in the period [impact_timestamp-0.5 s, impact_timestamp+0.5 s]
    '''
    if os.path.exists(path_result + 'Analyses/EventSeverity.csv'):
        print('--- Analysis 1: Event severity completed ---')
    else:
        ground_truth = event_meta[event_meta['duration_enough']]
        results = []
        for pretraining, encoder_name, cross_attention_name in zip(pretraining_list, encoder_name_list, cross_attention_name_list):
            safety_evaluation = read_evaluation('SSSE', path_events, pretraining, encoder_name, cross_attention_name)
            safety_evaluation = safety_evaluation[safety_evaluation['event_id'].isin(ground_truth.index.values)]
            safety_evaluation['impact_time'] = ground_truth.loc[safety_evaluation['event_id'].values, 'impact_timestamp'].values/1000
            safety_evaluation['period_start'] = safety_evaluation['impact_time'] - 0.5
            safety_evaluation['period_end'] = safety_evaluation['impact_time'] + 0.5
            safety_evaluation = safety_evaluation[(safety_evaluation['time']>=safety_evaluation['period_start'])&
                                                (safety_evaluation['time']<=safety_evaluation['period_end'])]
            avg_intensity = safety_evaluation.groupby(['event_id','target_id'])['intensity'].mean().reset_index().set_index('event_id')
            result = []
            for event_id in tqdm(avg_intensity.index.unique(), desc=pretraining+'_'+encoder_name+'_'+cross_attention_name):
                if ground_truth.loc[event_id, 'severity_first'] >= ground_truth.loc[event_id, 'severity_second']:
                    severity_higher = ground_truth.loc[event_id, 'severity_first']
                    severity_lower = ground_truth.loc[event_id, 'severity_second']
                else:
                    severity_higher = ground_truth.loc[event_id, 'severity_second']
                    severity_lower = ground_truth.loc[event_id, 'severity_first']
                if severity_higher>0 or severity_lower>0:
                    event_meta.loc[event_id, 'danger_present'] = True
                sorted_intensity = avg_intensity.loc[[event_id]].sort_values('intensity', ascending=False)
                intensity_higher = sorted_intensity.iloc[0]['intensity']
                intensity_lower = sorted_intensity.iloc[1]['intensity'] if len(sorted_intensity)>1 else 0.
                result.append([str(event_id)+'-higher', severity_higher, intensity_higher])
                result.append([str(event_id)+'-lower', severity_lower, intensity_lower])
            result = pd.DataFrame(result, columns=['event','given_severity','evaluated_intensity'])
            result['model'] = pretraining + '_' + encoder_name + '_' + cross_attention_name
            results.append(result)
        results = pd.concat(results)
        results.to_csv(path_result + f'Analyses/EventSeverity.csv', index=False)
        print('--- Analysis 1: Event severity completed ---')

    '''
    Analysis 2 - Conflict detection comparison
    For each event, it is applicable to compare conflict detection if both safety and danger are present
    - danger: period near impact_timestamp
              * start: after start_timestamp and before impact_timestamp
              * end: after impact_timestamp and before end_timestamp
    - safety: period 3 seconds before start_timestamp with 
              * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
              * not stopping, i.e., both ego and target speed > 0.5 m/s in the period
    The target has largest intensity/DRAC (or smallest TTC) during danger period is considered 
    as the conflicting target, and the safe period is determined specifically
    Then the comparison of ROC curves is between different indicators under various thresholds
    '''
    flag_to_compute = True
    if os.path.exists(path_result + 'Analyses/ConflictWarning.h5'):
        existing_results = pd.read_hdf(path_result + 'Analyses/ConflictWarning.h5', key='results')
        analysed_models = existing_results['model'].unique()
        ssse_models = [pretraining + '_' + encoder_name + '_' + cross_attention_name for pretraining, encoder_name, cross_attention_name in zip(pretraining_list, encoder_name_list, cross_attention_name_list)]
        if np.all(np.isin(['drac','ttc']+ssse_models, analysed_models)):
            print('--- Analysis 2: Conflict detection comparison completed ---')
            flag_to_compute = False
    if flag_to_compute:
        danger_start = np.minimum(event_meta['impact_timestamp'].values, event_meta['start_timestamp'].values)
        danger_end = np.minimum(event_meta['impact_timestamp'].values, event_meta['end_timestamp'].values)
        event_meta['danger_start'] = danger_start
        event_meta['danger_end'] = danger_end
        results = []

        sub_flag_to_compute = True
        if os.path.exists(path_result + 'Analyses/ConflictWarning.h5'):
            if 'drac' in analysed_models:
                results.append(existing_results[existing_results['model']=='drac'])
                print('--- Analysis 2 with DRAC already completed ---')
                sub_flag_to_compute = False
        if sub_flag_to_compute:
            drac_thresholds = np.round(np.arange(0.05,5.,0.05),2)
            safety_evaluation = read_evaluation('DRAC', path_events)
            progress_bar = tqdm(drac_thresholds, desc='DRAC', ascii=True, dynamic_ncols=False)
            drac_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'DRAC') for threshold in progress_bar)
            drac_records = pd.concat(drac_records).reset_index()
            drac_records['indicator'] = 'DRAC'
            drac_records['model'] = 'drac'
            results.append(drac_records)

        sub_flag_to_compute = True
        if os.path.exists(path_result + 'Analyses/ConflictWarning.h5'):
            if 'ttc' in analysed_models:
                results.append(existing_results[existing_results['model']=='ttc'])
                print('--- Analysis 2 with TTC already completed ---')
                sub_flag_to_compute = False
        if sub_flag_to_compute:
            ttc_thresholds = np.round(np.arange(0.2,20.,0.2),1)
            safety_evaluation = read_evaluation('TTC', path_events)
            progress_bar = tqdm(ttc_thresholds, desc='TTC', ascii=True, dynamic_ncols=False)
            ttc_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'TTC') for threshold in progress_bar)
            ttc_records = pd.concat(ttc_records).reset_index()
            ttc_records['indicator'] = 'TTC'
            ttc_records['model'] = 'ttc'
            results.append(ttc_records)

        ssse_thresholds = np.arange(2,101)
        for pretraining, encoder_name, cross_attention_name in zip(pretraining_list, encoder_name_list, cross_attention_name_list):
            sub_flag_to_compute = True
            model_name = pretraining + '_' + encoder_name + '_' + cross_attention_name
            if os.path.exists(path_result + 'Analyses/ConflictWarning.h5'):
                if model_name in analysed_models:
                    results.append(existing_results[existing_results['model']==model_name])
                    print('--- Analysis 2 with', model_name, 'already completed ---')
                    sub_flag_to_compute = False
            if sub_flag_to_compute:
                print('--- Analyzing', model_name, '---')
                safety_evaluation = read_evaluation('SSSE', path_events, pretraining, encoder_name, cross_attention_name)
                progress_bar = tqdm(ssse_thresholds, desc='SSSE', ascii=True, dynamic_ncols=False)
                ssse_records = Parallel(n_jobs=-1)(delayed(parallel_records)(threshold, safety_evaluation, event_data, event_meta[event_meta['duration_enough']], 'SSSE') for threshold in progress_bar)
                ssse_records = pd.concat(ssse_records).reset_index()
                ssse_records['indicator'] = 'SSSE'
                ssse_records['model'] = model_name
                results.append(ssse_records)

        results = pd.concat(results).reset_index(drop=True)
        results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
        results.loc[results['danger_evaluated'].isna(), 'danger_evaluated'] = False
        results.loc[results['safety_recorded'].isna(), 'safety_recorded'] = False
        results[['danger_recorded', 'danger_evaluated', 'safety_recorded']] = results[['danger_recorded', 'danger_evaluated', 'safety_recorded']].astype(bool)
        results.to_hdf(path_result + 'Analyses/ConflictWarning.h5', key='results', mode='w')
        print('--- Analysis 2: Conflict detection comparison completed ---')

    # '''
    # Analysis 3 - Warning timeliness
    #         - reaction_covered: at least one target presented in danger also presents 1 sec earlier than reaction_timestamp
    # '''
    #         target_covered = target_present[target_present['min']<=event_meta.loc[event_id, 'reaction_timestamp']/1000-1.]
    #         if len(target_covered)>0:
    #             event_meta.loc[event_id, 'reaction_covered'] = True
    #         else:
    #             event_meta.loc[event_id, 'reaction_covered'] = False

    # '''
    # Analysis 4 - Conflicting target identification
    # '''


    # event_meta.to_csv(path_result + 'Analyses/EventMeta.csv')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    path_prepared = './PreparedData/'
    path_result = './ResultData/'
    os.makedirs(path_result + 'Analyses/', exist_ok=True)

    main(path_prepared, path_result)
