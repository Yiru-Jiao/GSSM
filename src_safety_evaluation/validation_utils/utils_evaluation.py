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
    if indicator in ['TAdv', 'TTC2D', 'ACT', 'EI']:
        safety_evaluation = pd.read_hdf(path_results + f'TAdv_TTC2D_ACT_EI.h5', key='data')
        return safety_evaluation
    elif indicator=='UCD':
        safety_evaluation = pd.read_hdf(path_results + f'highD_UCD.h5', key='data')
        return safety_evaluation
    elif indicator=='GSSM':
        if np.any([config is None for config in [dataset_name, encoder_name, pretraining]]):
            print('Please specify model configuration for GSSM evaluation.')
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


def define_model(device, path_prepared, dataset, encoder_selection, pretrained_encoder, single_output=None, return_attention=False):
    # Define the model
    pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, pretrained_encoder, single_output, return_attention)
    # Load trained model
    pipeline.load_model()
    print(f'Model loaded: {pipeline.dataset_name}-{pipeline.encoder_name}-{pipeline.pretrained_encoder}')
    return pipeline.model


def lognormal_pdf(x, mu, sigma, rescale=True):
    p = 1/x/np.sqrt(2*np.pi)/sigma*np.exp(-1/2*(np.log(x)-mu)**2/sigma**2)
    if rescale:
        mode = np.exp(mu-sigma**2)
        pmax = 1/mode/np.sqrt(2*np.pi)/sigma*np.exp(-1/2*sigma**2)
        p = p/pmax
    return p


def lognormal_cdf(x, mu, sigma):
    x = np.maximum(1e-6, x)
    return 1/2+1/2*erf((np.log(x)-mu)/sigma/np.sqrt(2))


def extreme_cdf(x, mu, sigma, n=10):
    return (1-lognormal_cdf(x,mu,sigma))**n


def torch_intensity(spacing, mu, log_var=None, var=None):
    assert spacing.size() == mu.size(), f'{spacing.size()} != {mu.size()}'
    assert spacing.size() == log_var.size(), f'{spacing.size()} != {log_var.size()}'
    log_p = torch.log(torch.tensor(0.5))
    log_s = torch.log(torch.clamp(spacing, min=1e-6))
    if var is None:
        squared2var = torch.sqrt(2*torch.exp(log_var))
    elif log_var is None:
        squared2var = torch.sqrt(2*var)
    else:
        ValueError('At least one of log_var and var should be provided.')
    one_minus_cdf = 0.5*(1-torch.erf((log_s-mu)/squared2var))
    max_intensity = log_p / torch.log(torch.clamp(one_minus_cdf, min=1e-6, max=1-1e-6))
    return torch.log10(torch.clamp(max_intensity, min=1.))


def send_x_to_device(x, device):
    if isinstance(x, list):
        return tuple([i.to(device) for i in x])
    else:
        return x.to(device)


class custom_dataset(Dataset): 
    def __init__(self, X, s):
        self.X = X
        self.s = torch.from_numpy(s).float()
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
        return self.get_item(idx), self.s[idx]
    

def GSSM(states, model, device):
    contexts, spacing_list = states
    data_loader = DataLoader(custom_dataset(contexts, spacing_list), batch_size=1024, shuffle=False)

    mu_list = []
    sigma_list = []
    logn_list = []
    with torch.no_grad():
        for X, s in data_loader:
            out = model(send_x_to_device(X, device))
            mu, log_var = out
            logn = torch_intensity(s.to(device), mu, log_var=log_var)
            mu_list.append(mu.cpu().numpy()) # [n_samples]
            sigma_list.append(np.exp(0.5*log_var.cpu().numpy()))
            logn_list.append(logn.cpu().numpy())

    mu = np.concatenate(mu_list)
    sigma = np.concatenate(sigma_list)
    logn = np.concatenate(logn_list)

    return mu, sigma, logn # around (0.050171666, 693146.834) -> (0, 5.8408)


def determine_conflicts(evaluation, conflict_indicator, threshold):
    evaluation = evaluation.reset_index()
    evaluation['conflict'] = False

    if conflict_indicator in ['TAdv', 'TTC2D', 'ACT']:
        evaluation.loc[(evaluation[conflict_indicator]<threshold), 'conflict'] = True
        return evaluation
    
    elif conflict_indicator in ['EI']:
        evaluation.loc[(evaluation[conflict_indicator]>threshold), 'conflict'] = True
        return evaluation
    
    elif conflict_indicator=='GSSM':
        evaluation.loc[evaluation['intensity']>threshold, 'conflict'] = True
        evaluation['probability'] = extreme_cdf(evaluation['proximity'].values, evaluation['mu'].values, evaluation['sigma'].values, threshold)
        return evaluation


def is_target_recorded(danger, pre_danger, target_id, indicator):
    '''
    The conflicting target was not marked in the SHRP2 data, and it's possible the target was not recorded in
    the data. We thus need to filter out those invalid cases. 
    Later the method evaluation will assess missed detection based on [the target in danger] and assess false
    detection based on [other surrounding vehicles in the pre-danger period]. 
    To remain those eval-use data unintervened, this filtering is based on [the target in the pre-danger period] 
    and [other surrounding vehicles in the danger period]. It is thus assumed that the conflicting target is less
    risky in the pre-danger period than within the danger period. 
    More specifically,
    1) if the selected target has no data before the danger period, a filtering is not possible so we 
       skip the event, better being conservative than misleading;
    2) for the selected target, the 25th, 50th, and 75th percentiles of the risk indicated by the indicator values
       should be less in the pre-danger period than in the danger period, otherwise, the real conflicting target
       might be missed.
    * The use of 25th, 50th, and 75th percentiles (rather than e.g., min, mean, max) intends for a more robust 
      comparison and to avoid potential influence of outliers.
    '''
    danger = danger.loc[[target_id]]
    pre_danger = pre_danger[pre_danger['target_id']==target_id]

    # 1) No data before the danger period
    if len(pre_danger)<1:
        target_not_recorded = True
        percentiles_pre_danger = [np.nan, np.nan, np.nan]
        percentiles_danger = [np.nan, np.nan, np.nan]
    else:
        target_not_recorded = False
        # To avoid error in returning percentiles, replace inf with a large value and -inf with a small value
        pre_danger_positive_inf = np.isinf(pre_danger[indicator])&(pre_danger[indicator]>0)
        pre_danger_negative_inf = np.isinf(pre_danger[indicator])&(pre_danger[indicator]<0)
        danger_positive_inf = np.isinf(danger[indicator])&(danger[indicator]>0)
        danger_negative_inf = np.isinf(danger[indicator])&(danger[indicator]<0)
        pre_danger.loc[pre_danger_positive_inf, indicator] = max(1e6, pre_danger.loc[~pre_danger_positive_inf, indicator].max())
        pre_danger.loc[pre_danger_negative_inf, indicator] = min(-1e6, pre_danger.loc[~pre_danger_negative_inf, indicator].min())
        danger.loc[danger_positive_inf, indicator] = max(1e6, danger.loc[~danger_positive_inf, indicator].max())
        danger.loc[danger_negative_inf, indicator] = min(-1e6, danger.loc[~danger_negative_inf, indicator].min())
        if indicator in ['TAdv', 'TTC2D', 'ACT']:
            # For these indicators, the smaller the value, the higher the risk
            percentiles_pre_danger = pre_danger[indicator].quantile([0.25, 0.5, 0.75]).values
            percentiles_danger = danger[indicator].quantile([0.25, 0.5, 0.75]).values
            for i in range(3):
                if percentiles_pre_danger[i] <= percentiles_danger[i]:
                    target_not_recorded = True
        elif indicator in ['intensity', 'EI']:
            # For these indicators, the larger the value, the higher the risk
            percentiles_pre_danger = pre_danger[indicator].quantile([0.25, 0.5, 0.75]).values
            percentiles_danger = danger[indicator].quantile([0.25, 0.5, 0.75]).values
            for i in range(3):
                if percentiles_pre_danger[i] >= percentiles_danger[i]:
                    target_not_recorded = True

    return target_not_recorded, (percentiles_pre_danger, percentiles_danger)


def determine_target(indicator, danger, pre_danger):
    if indicator == 'GSSM':
        indicator = 'intensity'
    if len(danger)<1: # no surrounding vehicles recorded in the danger period
        target_id = np.nan
        indicator_values = ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan])
    elif danger[indicator].isna().all() or pre_danger[indicator].isna().all(): # in case of invalid values
        target_id = np.nan
        indicator_values = ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan])
    else: # a target is temporarily selected as the most risky one in the danger period
        if indicator in ['TAdv', 'TTC2D', 'ACT']:
            target_id = danger.groupby('target_id')[indicator].mean().idxmin()
            target_not_recorded, indicator_values = is_target_recorded(danger, pre_danger, target_id, indicator)
        elif indicator in ['EI', 'intensity']:
            target_id = danger.groupby('target_id')[indicator].mean().idxmax()
            target_not_recorded, indicator_values = is_target_recorded(danger, pre_danger, target_id, indicator)

        if target_not_recorded:
            target_id = np.nan
            indicator_values = ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan])
    return target_id, indicator_values


def parallel_records(threshold, safety_evaluation, event_data, event_meta, indicator):
    safety_evaluation = safety_evaluation.sort_values(['target_id','time']).set_index('event_id')
    event_data = event_data.reset_index().set_index(['event_id', 'target_id', 'time'])
    event_meta = event_meta[event_meta['duration_enough']&(event_meta['conflict']!='none')]
    event_ids = np.intersect1d(event_meta.index.values, safety_evaluation.index.unique())

    records = event_meta[['danger_start', 'danger_end']].copy()
    initial_columns = ['danger_recorded', 'true_warning', 'safety_recorded', 'safe_target_ids', 'num_false_warning', 'num_true_non_warning', 'false_warning']
    initial_vlaues = [False, np.nan, False, 'none', np.nan, np.nan, np.nan]
    for event_id in event_ids:
        event = safety_evaluation.loc[event_id]
        records.loc[event_id, initial_columns] = initial_vlaues

        danger = event[(event['time']>=event_meta.loc[event_id, 'danger_start']/1000)&
                       (event['time']<=event_meta.loc[event_id, 'danger_end']/1000)].reset_index().set_index('target_id')
        pre_danger = event[(event['time']<event_meta.loc[event_id, 'danger_start']/1000)].reset_index()

        # Determine the conflicting target and warning
        target_id, indicator_values = determine_target(indicator, danger, pre_danger)
        if np.isnan(target_id):
            continue
        target_danger = danger.loc[[target_id]]
        if len(target_danger)<20:
            # a potential target in danger for at least 2 seconds
            continue
        records.loc[event_id, 'danger_recorded'] = True
        records.loc[event_id, 'target_id'] = target_id
        records.loc[event_id, ['25th_pre_danger','50th_pre_danger','75th_pre_danger']] = indicator_values[0]
        records.loc[event_id, ['25th_danger','50th_danger','75th_danger']] = indicator_values[1]
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
        * end: considering 2~5 seconds after start, at least 3 seconds before conflict start_timestamp
        '''
        targets = event[event['target_id']!=target_id]
        targets_period = targets[targets['time']<(event_meta.loc[event_id, 'start_timestamp']/1000-3.)]
        if len(targets_period) < 35:
            # later there will be a filter of 35 time moments per target, so here it is meaningless if less than 35
            records.loc[event_id, 'safety_recorded'] = False
            continue
        safety_recorded = False
        target_ids = []
        false_warnings = 0
        targets_period = targets_period.reset_index().set_index(['event_id', 'target_id', 'time'])
        for target_id in targets_period.index.get_level_values('target_id').unique():
            target_period = targets_period.loc[(event_id, target_id)]
            if len(target_period)<35:
                # a potential target in safety for at least 2 seconds after start (i.e. 15+20)
                continue
            target_period = target_period.iloc[15:65]
            no_hard_braking = (event_data.loc[(event_id, target_id, target_period.index), 'acc_ego'].min()>-1.5)
            if no_hard_braking:
                safety_recorded = True
                target_ids.append(target_id)
                target_period = determine_conflicts(target_period, indicator, threshold)
                if np.any(target_period['conflict']):
                    false_warnings += 1
        true_non_warnings = len(target_ids) - false_warnings

        if safety_recorded:
            records.loc[event_id, ['safety_recorded', 'num_false_warning', 'num_true_non_warning']] = [True, false_warnings, true_non_warnings]
            records.loc[event_id, 'safe_target_ids'] = ','.join([str(i) for i in target_ids])
            records.loc[event_id, 'false_warning'] = 1 if false_warnings>0 else 0

    records['threshold'] = threshold
    return records


def optimize_threshold(warning, conflict_indicator, curve_type, return_stats=False):
    # true_positives = warning[warning['danger_recorded']&(warning['true_warning']>0.5)].groupby('threshold').size()
    # false_positives = warning[warning['safety_recorded']&(warning['false_warning']>0.5)].groupby('threshold').size()
    # true_negatives = warning[warning['safety_recorded']&(warning['false_warning']<0.5)].groupby('threshold').size()
    # false_negatives = warning[warning['danger_recorded']&(warning['true_warning']<0.5)].groupby('threshold').size()
    true_positives = warning[warning['danger_recorded']&(warning['true_warning']>0.5)].groupby('threshold').size()
    false_positives = warning[warning['safety_recorded']].groupby('threshold')['num_false_warning'].sum()
    true_negatives = warning[warning['safety_recorded']].groupby('threshold')['num_true_non_warning'].sum()
    false_negatives = warning[warning['danger_recorded']&(warning['true_warning']<0.5)].groupby('threshold').size()
    statistics = pd.concat([true_positives, false_positives, true_negatives, false_negatives], axis=1, keys=['TP', 'FP', 'TN', 'FN'])
    statistics = statistics.fillna(0).reset_index().sort_values('threshold') # nan can be caused by empty combination of threshold and warning

    if curve_type=='ROC':
        statistics['false negative rate'] = statistics['FN']/(statistics['TP']+statistics['FN'])
        statistics['false positive rate'] = statistics['FP']/(statistics['FP']+statistics['TN'])
        if conflict_indicator in ['TAdv', 'TTC2D', 'ACT']:
            statistics = statistics.sort_values(by=['false positive rate','false negative rate','threshold'], ascending=[True, True, True])
        elif conflict_indicator in ['GSSM', 'EI']:
            statistics = statistics.sort_values(by=['false positive rate','false negative rate','threshold'], ascending=[True, True, False])
        statistics['combined rate'] = statistics['false negative rate']**2+statistics['false positive rate']**2
    elif curve_type=='PRC':
        statistics['precision'] = statistics['TP']/(statistics['TP']+statistics['FP'])
        statistics['recall'] = statistics['TP']/(statistics['TP']+statistics['FN'])
        if conflict_indicator in ['TAdv', 'TTC2D', 'ACT']:
            statistics = statistics.sort_values(by=['recall','precision','threshold'], ascending=[False, False, True])
        elif conflict_indicator in ['GSSM', 'EI']:
            statistics = statistics.sort_values(by=['recall','precision','threshold'], ascending=[False, False, False])
        statistics['combined rate'] = (1-statistics['recall'])**2+(1-statistics['precision'])**2
    statistics['combined rate'] = np.round(np.sqrt(statistics['combined rate']), 2)
    optimal_rate = statistics['combined rate'].min()
    if np.isnan(optimal_rate):
        if return_stats:
            return statistics, None, None
        else:
            return np.nan
    else:
        optimal_warning = statistics.loc[statistics['combined rate']==optimal_rate]
        optimal_threshold = optimal_warning.iloc[0]
        if return_stats:
            optimal_warning = warning[warning['threshold']==optimal_threshold['threshold']]
            return statistics, optimal_warning, optimal_threshold
        else:
            if curve_type=='ROC':
                print(warning['model'].values[0], ' ', conflict_indicator, ' ', curve_type,
                    ' optimal threshold: ', optimal_threshold['threshold'], 
                    ' false negative rate: ', round(optimal_threshold['false negative rate']*100, 2),
                    ' false positive rate: ', round(optimal_threshold['false positive rate']*100, 2))
            elif curve_type=='PRC':
                print(warning['model'].values[0], ' ', conflict_indicator, ' ', curve_type,
                    ' optimal threshold: ', optimal_threshold['threshold'], 
                    ' precision: ', round(optimal_threshold['precision']*100, 2),
                    ' recall: ', round(optimal_threshold['recall']*100, 2))
            return optimal_threshold['threshold']


def issue_warning(indicator, optimal_threshold, safety_evaluation, event_meta):
    safety_evaluation = safety_evaluation.sort_values(['target_id','time']).set_index('event_id')
    event_meta = event_meta[event_meta['duration_enough']&(event_meta['conflict']!='none')]
    event_ids = np.intersect1d(event_meta.index.values, safety_evaluation.index.unique())

    records = event_meta[['danger_start', 'danger_end', 'reaction_timestamp', 'impact_timestamp']].copy()
    records['indicator'] = indicator
    records['optimal_threshold'] = optimal_threshold
    records['danger_recorded'] = False
    for event_id in event_ids:
        event = safety_evaluation.loc[event_id].reset_index().set_index('target_id')

        danger = event[(event['time']>=event_meta.loc[event_id, 'danger_start']/1000)&
                       (event['time']<=event_meta.loc[event_id, 'danger_end']/1000)]
        pre_danger = event[(event['time']<event_meta.loc[event_id, 'danger_start']/1000)].reset_index()

        # Determine the conflicting target
        target_id, indicator_values = determine_target(indicator, danger, pre_danger)
        if np.isnan(target_id):
            continue
        target_danger = danger.loc[[target_id]]
        if len(target_danger)<20:
            # a potential target in danger for at least 2 seconds
            continue
        records.loc[event_id, 'danger_recorded'] = True
        records.loc[event_id, 'target_id'] = target_id
        records.loc[event_id, ['25th_pre_danger','50th_pre_danger','75th_pre_danger']] = indicator_values[0]
        records.loc[event_id, ['25th_danger','50th_danger','75th_danger']] = indicator_values[1]
        target = determine_conflicts(event.loc[target_id], indicator, optimal_threshold)

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
