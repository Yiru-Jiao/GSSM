'''
This script contains utility functions for safety evaluation and analysis.
'''

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.special import erf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test


def read_events(path_events, meta_only=False):
    event_categories = sorted(os.listdir(path_events))
    event_meta = pd.concat([pd.read_csv(path_events + f'{event_cat}/event_meta.csv') for event_cat in event_categories])
    event_meta = event_meta.set_index('event_id')
    if meta_only:
        return event_meta
    else:
        event_data = pd.concat([pd.read_hdf(path_events + f'{event_cat}/event_data.h5', key='data') for event_cat in event_categories])
        assert np.all(np.isin(event_data['event_id'].unique(), event_meta.index.values))
        return event_meta, event_data


def read_evaluation(indicator, path_results, dataset_name=None, encoder_name=None, pretraining=None):
    if indicator in ['TTC', 'DRAC', 'MTTC', 'PSD']:
        safety_evaluation = pd.read_hdf(path_results + f'TTC_DRAC_MTTC_PSD.h5', key='data')
        return safety_evaluation
    elif indicator in ['TAdv', 'TTC2D', 'ACT', 'EI']:
        safety_evaluation = pd.read_hdf(path_results + f'TAdv_TTC2D_ACT_EI.h5', key='data')
        return safety_evaluation
    elif indicator=='UCD':
        safety_evaluation = pd.read_hdf(path_results + f'highD_UCD.h5', key='data')
        return safety_evaluation
    elif indicator=='SSSE':
        if np.any([config is None for config in [dataset_name, encoder_name, pretraining]]):
            print('Please specify model configuration for SSSE evaluation.')
            return None
        else:
            safety_evaluation = pd.read_hdf(path_results + f'{dataset_name}_{encoder_name}_{pretraining}.h5', key='data')
            return safety_evaluation


def set_veh_dimensions(event_meta, avg_width, avg_length):
    veh_dimensions = event_meta[['ego_width','ego_length','target_width','target_length']].copy()
    condition = event_meta[['target_width','target_length']].isna().any(axis=1)
    veh_dimensions.loc[condition, ['target_width','target_length']] = event_meta.loc[condition, ['other_width','other_length']].values
    for var in ['ego_width','ego_length','target_width','target_length']:
        veh_dimensions.loc[veh_dimensions[var].isna(), var] = avg_width if 'width' in var else avg_length
    return veh_dimensions


def define_model(device, path_prepared, dataset, encoder_selection, pretrained_encoder, return_attention=False):
    # Define the model
    pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, pretrained_encoder, return_attention)
    ## Load trained model
    pipeline.load_model()
    print(f'Model loaded: {pipeline.dataset_name}-{pipeline.encoder_name}')
    return pipeline.model


def lognormal_pdf(x, mu, sigma, rescale=True):
    p = 1/x/np.sqrt(2*np.pi)/sigma*np.exp(-1/2*(np.log(x)-mu)**2/sigma**2)
    if rescale:
        mode = np.exp(mu-sigma**2)
        pmax = 1/mode/np.sqrt(2*np.pi)/sigma*np.exp(-1/2*sigma**2)
        p = p/pmax
    return p


def lognormal_cdf(x, mu, sigma):
    return 1/2+1/2*erf((np.log(x)-mu)/sigma/np.sqrt(2))


def extreme_cdf(x, mu, sigma, n=10):
    return (1-lognormal_cdf(x,mu,sigma))**n


def send_x_to_device(x, device):
    if isinstance(x, tuple):
        return tuple([i.to(device) for i in x])
    else:
        return x.to(device)


class custom_dataset(Dataset): 
    def __init__(self, X):
        self.X = X
        if isinstance(X, tuple):
            def get_length():
                return len(self.X[0])
            def get_item(idx):
                return tuple([torch.from_numpy(x_i[idx]).float() for x_i in self.X])
        else:
            def get_length():
                return len(self.X)
            def get_item(idx):
                return torch.from_numpy(self.X[idx]).float()
        self.get_length = get_length
        self.get_item = get_item

    def __len__(self): 
        return self.get_length()

    def __getitem__(self, idx): 
        return self.get_item(idx)
    

def SSSE(states, model, device):
    contexts, spacing_list = states
    data_loader = DataLoader(custom_dataset(contexts), batch_size=1024, shuffle=False)

    mu_list = []
    sigma_list = []
    with torch.no_grad():
        for x in data_loader:
            out = model(send_x_to_device(x, device))
            mu, log_var = out
            mu_list.append(mu.cpu().numpy()) # [n_samples]
            sigma_list.append(np.exp(0.5*log_var.cpu().numpy()))

    mu = np.concatenate(mu_list)
    sigma = np.concatenate(sigma_list)

    # 0.5 means that the probability of conflict is larger than the probability of non-conflict
    max_intensity = np.log(0.5)/np.log(1-lognormal_cdf(spacing_list, mu_list, sigma_list)+1e-6)

    return mu, sigma, max_intensity


def determine_conflicts(evaluation, conflict_indicator, threshold):
    evaluation = evaluation.reset_index()
    evaluation['conflict'] = False

    if conflict_indicator in ['TTC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT']:
        evaluation.loc[(evaluation[conflict_indicator]<threshold), 'conflict'] = True
        return evaluation
    
    elif conflict_indicator in ['DRAC', 'EI']:
        evaluation.loc[(evaluation[conflict_indicator]>threshold), 'conflict'] = True
        if conflict_indicator=='DRAC':
            evaluation.loc[(evaluation['DRAC']<0), 'conflict'] = True # DRAC<0 means overlapping bounding boxes
        return evaluation
    
    elif conflict_indicator=='SSSE':
        evaluation['probability'] = extreme_cdf(evaluation['proximity'].values, evaluation['mu'].values, evaluation['sigma'].values, threshold)
        evaluation.loc[evaluation['intensity']>threshold, 'conflict'] = True
        return evaluation


def determine_target(indicator, danger, before_danger):
    if indicator == 'SSSE':
        indicator = 'intensity'
    if danger[indicator].isna().all() or before_danger[indicator].isna().all():
        target_id = np.nan
        median_before_danger = np.nan
        median_danger = np.nan
    else:
        if indicator in ['TTC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT']:
            target_id = danger.loc[danger[indicator].idxmin(),'target_id']
        elif indicator in ['DRAC', 'EI', 'intensity']:
            target_id = danger.loc[danger[indicator].idxmax(),'target_id']
        median_before_danger = before_danger[before_danger['target_id']!=target_id][indicator].median()
        median_danger = danger[danger['target_id']==target_id][indicator].median()
    return target_id, median_before_danger, median_danger


def parallel_records(threshold, safety_evaluation, event_data, event_meta, indicator):
    safety_evaluation = safety_evaluation.sort_values(['target_id','time']).set_index('event_id')
    event_data = event_data.reset_index().set_index(['event_id', 'target_id', 'time'])
    event_meta = event_meta[event_meta['duration_enough']]
    event_ids = np.intersect1d(event_meta.index.values, safety_evaluation.index.unique())

    records = event_meta[['danger_start', 'danger_end']].copy()
    for event_id in event_ids:
        event = safety_evaluation.loc[event_id].copy()

        danger = event[(event['time']>=event_meta.loc[event_id, 'danger_start']/1000)&
                       (event['time']<=event_meta.loc[event_id, 'danger_end']/1000)].reset_index()
        before_danger = event[(event['time']<event_meta.loc[event_id, 'danger_start']/1000)].reset_index()
        if danger.groupby('target_id')['time'].count().max()<35: # a potential target should appear at least 3.5 seconds
            records.loc[event_id, 'danger_recorded'] = False
            continue

        # Determine the conflicting target and warning
        target_id, median_before_danger, median_danger = determine_target(indicator, danger, before_danger)
        records.loc[event_id, 'target_id'] = target_id
        records.loc[event_id, 'median_before_danger'] = median_before_danger
        records.loc[event_id, 'median_danger'] = median_danger
        if np.isnan(target_id):
            records.loc[event_id, 'danger_recorded'] = False
            continue
        target_danger = danger[danger['target_id']==target_id]
        records.loc[event_id, 'danger_recorded'] = True
        target_danger = determine_conflicts(target_danger, indicator, threshold)
        if target_danger['conflict'].sum()>5: # at least warning for 0.5 second
            records.loc[event_id, 'true_warning'] = 1
        else:
            records.loc[event_id, 'true_warning'] = 0

        # Determine safety period for the conflicting target
        '''
        the beginning of a non-target vehicle in an event before start_timestamp with no hard braking
        * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
        * start: 1.5 seconds after the target is detected
        * end: considering 5 seconds after start, at least 3 seconds before conflict start_timestamp
        '''
        targets = event[event['target_id']!=target_id]
        targets_period = targets[targets['time']<(event_meta.loc[event_id, 'start_timestamp']/1000-3.)]
        if len(targets_period) <= 0:
            records.loc[event_id, 'safety_recorded'] = False
            continue
        safety_recorded = False
        target_ids = []
        false_warnings = 0
        for target_id in targets_period['target_id'].unique():
            target_period = targets_period[targets_period['target_id']==target_id]
            if len(target_period)<35:
                continue
            else:
                target_period = target_period.iloc[15:65]
                motion_states = ['acc_ego','v_ego','v_sur']
                multi_index = pd.MultiIndex.from_arrays([target_period.index.values,
                                                         target_period['target_id'].values,
                                                         target_period['time'].values], names=('event_id','target_id','time'))
                target_period[motion_states] = event_data.loc[multi_index, motion_states].values
                no_hard_braking = (target_period['acc_ego'].min()>-1.5)
                if no_hard_braking:
                    safety_recorded = True
                    target_ids.append(target_id)
                    target_period = determine_conflicts(target_period.copy(), indicator, threshold)
                    if np.any(target_period['conflict']):
                        false_warnings += 1

        if safety_recorded:
            records.loc[event_id, 'safety_recorded'] = True
            records.loc[event_id, 'false_warning'] = false_warnings
            records.loc[event_id, 'true_non_warning'] = len(target_ids) - false_warnings
            records.loc[event_id, 'safe_target_ids'] = ','.join([str(i) for i in target_ids])
        else:
            records.loc[event_id, 'safety_recorded'] = False

    records['threshold'] = threshold
    records['safe_target_ids'] = records['safe_target_ids'].fillna('none')
    return records


def optimize_threshold(warning, conflict_indicator, curve_type, return_stats=False):
    if conflict_indicator in ['TTC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT']:
        warning.loc[warning['median_before_danger']<warning['median_danger'], 'danger_recorded'] = False
    elif conflict_indicator in ['DRAC', 'EI', 'SSSE']:
        warning.loc[warning['median_before_danger']>warning['median_danger'], 'danger_recorded'] = False

    true_positives = warning[warning['danger_recorded']&(warning['true_warning']>0.5)].groupby('threshold').size()
    false_positives = warning[warning['safety_recorded']].groupby('threshold')['false_warning'].sum()
    true_negatives = warning[warning['safety_recorded']].groupby('threshold')['true_non_warning'].sum()
    false_negatives = warning[warning['danger_recorded']&(warning['true_warning']<0.5)].groupby('threshold').size()
    statistics = pd.concat([true_positives, false_positives, true_negatives, false_negatives], axis=1, keys=['TP', 'FP', 'TN', 'FN'])
    statistics = statistics.fillna(0).reset_index() # nan can be caused by empty combination of threshold and warning

    if curve_type=='ROC':
        statistics['true positive rate'] = statistics['TP']/(statistics['TP']+statistics['FN'])
        statistics['false positive rate'] = statistics['FP']/(statistics['FP']+statistics['TN'])
        if conflict_indicator in ['TTC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT']:
            statistics = statistics.sort_values(by=['false positive rate','true positive rate','threshold'], ascending=[True, False, True])
        else:
            statistics = statistics.sort_values(by=['false positive rate','true positive rate','threshold'], ascending=[True, False, False])
        statistics['combined rate'] = (1-statistics['true positive rate'])**2+(statistics['false positive rate'])**2
    elif curve_type=='PRC':
        statistics['precision'] = statistics['TP']/(statistics['TP']+statistics['FP'])
        statistics['recall'] = statistics['TP']/(statistics['TP']+statistics['FN'])
        if conflict_indicator in ['TTC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT']:
            statistics = statistics.sort_values(by=['recall','precision','threshold'], ascending=[False, False, True])
        else:
            statistics = statistics.sort_values(by=['recall','precision','threshold'], ascending=[False, False, False])
        statistics['combined rate'] = (1-statistics['recall'])**2+(1-statistics['precision'])**2
    optimal_rate = statistics['combined rate'].min()
    optimal_warning = statistics.loc[statistics['combined rate']==optimal_rate]
    optimal_threshold = optimal_warning.iloc[0]
    if return_stats:
        optimal_warning = warning[warning['threshold']==optimal_threshold['threshold']]
        return statistics, optimal_warning, optimal_threshold
    else:
        if curve_type=='ROC':
            print(warning['model'].values[0], ' ', conflict_indicator, ' ', curve_type,
                 ' optimal threshold: ', optimal_threshold['threshold'], 
                 ' true positive rate: ', round(optimal_threshold['true positive rate']*100, 2),
                 ' false positive rate: ', round(optimal_threshold['false positive rate']*100, 2))
        elif curve_type=='PRC':
            print(warning['model'].values[0], ' ', conflict_indicator, ' ', curve_type,
                 ' optimal threshold: ', optimal_threshold['threshold'], 
                 ' precision: ', round(optimal_threshold['precision']*100, 2),
                 ' recall: ', round(optimal_threshold['recall']*100, 2))
        return optimal_threshold['threshold']


def issue_warning(indicator, optimal_threshold, safety_evaluation, event_meta):
    safety_evaluation = safety_evaluation.sort_values(['target_id','time']).set_index('event_id')
    event_meta = event_meta[event_meta['duration_enough']]
    event_ids = np.intersect1d(event_meta.index.values, safety_evaluation.index.unique())

    records = event_meta[['danger_start', 'danger_end', 'reaction_timestamp', 'impact_timestamp']].copy()
    records['indicator'] = indicator
    records['optimal_threshold'] = optimal_threshold
    for event_id in event_ids:
        event = safety_evaluation.loc[event_id].copy()

        danger = event[(event['time']>=event_meta.loc[event_id, 'danger_start']/1000)&
                       (event['time']<=event_meta.loc[event_id, 'danger_end']/1000)].reset_index()
        before_danger = event[(event['time']<event_meta.loc[event_id, 'danger_start']/1000)].reset_index()
        if danger.groupby('target_id')['time'].count().max()<35:
            records.loc[event_id, 'danger_recorded'] = False
            continue

        # Determine the conflicting target
        target_id, median_before_danger, median_danger = determine_target(indicator, danger, before_danger)
        records.loc[event_id, 'target_id'] = target_id
        if np.isnan(target_id):
            records.loc[event_id, 'danger_recorded'] = False
            continue
        if (indicator in ['TTC', 'MTTC', 'PSD', 'TAdv', 'TTC2D', 'ACT']) and (median_before_danger < median_danger):
            records.loc[event_id, 'danger_recorded'] = False
            continue
        if (indicator in ['DRAC', 'EI', 'SSSE']) and (median_before_danger > median_danger):
            records.loc[event_id, 'danger_recorded'] = False
            continue
        target = event[event['target_id']==target_id]
        target = determine_conflicts(target, indicator, optimal_threshold)

        # Locate the first warning moment: the last safe->unsafe transition moment before impact timestamp
        warning = target[target['time']<=event_meta.loc[event_id, 'impact_timestamp']/1000]['conflict'].astype(int).values
        warning_change = warning[1:] - warning[:-1] # 1: safe->unsafe, -1: unsafe->safe, 0: no change
        first_warning = np.where(warning_change>0.5)[0]
        if len(first_warning)>0:
            records.loc[event_id,'first_warning_timestamp'] = target.iloc[first_warning[-1]+1]['time']*1000
        else: # no warning before impact
            records.loc[event_id,'first_warning_timestamp'] = np.nan

        # Calculate the warning period: the percentage of warning time moments within [danger_start, danger_end]
        target_danger = target[(target['time']>=event_meta.loc[event_id, 'danger_start']/1000)&
                               (target['time']<=event_meta.loc[event_id, 'danger_end']/1000)]
        true_warning = target_danger[target_danger['conflict']]
        records.loc[event_id, 'true_warning_period'] = len(true_warning) / len(target_danger)

    return records


# def encode_get_attention(states, model, device):
#     contexts, spacing_list = states
#     data_loader = DataLoader(custom_dataset(contexts), batch_size=1024, shuffle=False)

#     mu_list = []
#     sigma_list = []
#     attended_states = []
#     attention_matrices = {}
#     with torch.no_grad():
#         for x in data_loader:
#             out = model(send_x_to_device(x, device))
#             mu, log_var, hidden_states = out
#         mu_list.append(mu.cpu().numpy()) # (n_samples, 1)
#         sigma_list.append(np.exp(0.5*log_var.cpu().numpy()))
#         attended_states.append(hidden_states[0].cpu().numpy())
#         attention_matrices = {key: value.cpu().numpy() for key, value in hidden_states[1].items()}

#     mu = np.concatenate(mu_list, axis=0)
#     sigma = np.concatenate(sigma_list, axis=0)

#     return mu, sigma, attended_states, attention_matrices
