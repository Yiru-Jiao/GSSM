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
    event_categories = [event_cat for event_cat in event_categories if os.path.isdir(path_events + event_cat)]
    event_meta = pd.concat([pd.read_csv(path_events + f'{event_cat}/event_meta.csv') for event_cat in event_categories])
    event_meta = event_meta.set_index('event_id')
    if meta_only:
        return event_meta
    else:
        event_data = pd.concat([pd.read_hdf(path_events + f'{event_cat}/event_data.h5', key='data') for event_cat in event_categories])
        assert np.all(np.isin(event_data['event_id'].unique(), event_meta.index.values))
        return event_meta, event_data


def read_evaluation(indicator, path_results, dataset_name=None, encoder_name=None, cross_attention_name=None, pretraining=None):
    if indicator=='TTC' or indicator=='DRAC' or indicator=='MTTC':
        safety_evaluation = pd.read_hdf(path_results + f'TTC_DRAC_MTTC.h5', key='data')
        return safety_evaluation
    elif indicator=='SSSE':
        if np.any([config is None for config in [dataset_name, encoder_name, cross_attention_name, pretraining]]):
            print('Please specify model configuration for SSSE evaluation.')
            return None
        else:
            safety_evaluation = pd.read_hdf(path_results + f'{dataset_name}_{encoder_name}_{cross_attention_name}_{pretraining}.h5', key='data')
            return safety_evaluation


def define_model(device, path_prepared, dataset, encoder_selection, cross_attention, pretrained_encoder):
    # Define the model
    pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, cross_attention, pretrained_encoder, return_attention=True)
    ## Load trained model
    pipeline.load_model()
    print(f'Model loaded: {pipeline.dataset_name}-{pipeline.encoder_name}-{pipeline.cross_attention_name}')
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
    if isinstance(x, list):
        return [i.to(device) for i in x]
    else:
        return x.to(device)


class custom_dataset(Dataset): 
    def __init__(self, X):
        self.X = X
        if isinstance(X, tuple):
            def get_length():
                return len(self.X[0])
            def get_item(idx):
                return [torch.from_numpy(x_i[idx]).float() for x_i in self.X]
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
    

def SSSE(states, model, device, relative_angle):
    contexts, proximity = states
    data_loader = DataLoader(custom_dataset(contexts), batch_size=1024, shuffle=False)

    mu_list = []
    sigma_list = []
    for x in data_loader:
        with torch.no_grad():
            out = model(send_x_to_device(x, device))
            mu, sigma, _ = out
        mu_list.append(mu.squeeze().cpu().numpy())
        sigma_list.append(sigma.squeeze().cpu().numpy())

    mu = np.concatenate(mu_list, axis=0)
    sigma = np.concatenate(sigma_list, axis=0)

    # Modify mu when ego and sur are leaving each other
    leaving = (relative_angle<0)
    mu[leaving] = proximity[leaving]

    # 0.5 means that the probability of conflict is larger than the probability of non-conflict
    max_intensity = np.log(0.5)/np.log(1-lognormal_cdf(proximity, mu, sigma)+1e-6)
    max_intensity = np.maximum(1., max_intensity)

    return mu, sigma, max_intensity


def determine_conflicts(evaluation, conflict_indicator, threshold):
    evaluation = evaluation.reset_index()
    evaluation['conflict'] = False

    if conflict_indicator=='TTC' or conflict_indicator=='MTTC':
        evaluation.loc[(evaluation[conflict_indicator]<threshold), 'conflict'] = True
        return evaluation
    
    elif conflict_indicator=='DRAC':
        evaluation.loc[(evaluation['DRAC']>threshold), 'conflict'] = True
        return evaluation
    
    elif conflict_indicator=='SSSE':
        evaluation['probability'] = extreme_cdf(evaluation['proximity'].values, evaluation['mu'].values, evaluation['sigma'].values, threshold)
        # 0.5 means that the probability of conflict is larger than the probability of non-conflict
        evaluation.loc[evaluation['probability']>0.5, 'conflict'] = True
        return evaluation


def determine_target(indicator, danger):
    if indicator=='TTC' or indicator=='MTTC':
        if danger[indicator].isna().all():
            target_id = np.nan
        else:
            target_id = danger.loc[danger[indicator].idxmin(),'target_id']
    elif indicator=='DRAC':
        if danger['DRAC'].isna().all():
            target_id = np.nan
        else:
            target_id = danger.loc[danger['DRAC'].idxmax(),'target_id']
    elif indicator=='SSSE':
        if danger['intensity'].isna().all():
            target_id = np.nan
        else:
            target_id = danger.loc[danger['intensity'].idxmax(),'target_id']
    return target_id


def parallel_records(threshold, safety_evaluation, event_data, event_meta, indicator):
    event_data = event_data.reset_index().set_index(['event_id', 'target_id', 'time'])
    safety_evaluation = safety_evaluation.sort_values(['target_id','time'])
    events = safety_evaluation.set_index('event_id')
    event_ids = np.intersect1d(event_meta.index.values, events.index.unique())

    records = event_meta[['danger_start', 'danger_end']].copy()
    for event_id in event_ids:
        event = events.loc[event_id].copy()
        danger = event[(event['time']>=event_meta.loc[event_id, 'danger_start']/1000)&
                       (event['time']<=event_meta.loc[event_id, 'danger_end']/1000)].reset_index()
        if danger.groupby('target_id')['time'].count().max()<5:
            records.loc[event_id, 'danger_recorded'] = False
            continue

        # Determine the conflicting target and warning
        target_id = determine_target(indicator, danger)
        records.loc[event_id, 'target_id'] = target_id
        if np.isnan(target_id):
            records.loc[event_id, 'danger_recorded'] = False
            continue
        target_danger = danger[danger['target_id']==target_id]
        records.loc[event_id, 'danger_recorded'] = True
        records.loc[event_id, 'danger_period'] = len(target_danger)/10
        target_danger = determine_conflicts(target_danger, indicator, threshold)
        if np.any(target_danger['conflict']):
            records.loc[event_id, 'true warning'] = 1
        else:
            records.loc[event_id, 'true warning'] = 0

        # Determine safety period for the conflicting target
        '''
        the beginning in an event before start_timestamp with no hard braking
        * no hard braking, i.e., acceleration > -1.5 m/s^2 in the period
        * start: first evaluatable timestamp in the event
        * end: 0.5~5 seconds after the first timestamp, at least 3 seconds before start_timestamp
        '''
        target = event[event['target_id']==target_id]
        target_period = target[target['time']<(event_meta.loc[event_id, 'start_timestamp']/1000-3.)].copy()
        if len(target_period)<5:
            records.loc[event_id, 'safety_recorded'] = False
            continue
        target_period = target_period.iloc[:55]
        motion_states = ['acc_ego','v_ego','v_sur']
        multi_index = pd.MultiIndex.from_arrays([target_period.index.values,
                                                 target_period['target_id'].values,
                                                 target_period['time'].values], names=('event_id','target_id','time'))
        target_period[motion_states] = event_data.loc[multi_index, motion_states].values
        records.loc[event_id, 'avg_acc_ego'] = target_period['acc_ego'].mean()
        records.loc[event_id, 'avg_v_ego'] = target_period['v_ego'].mean()
        records.loc[event_id, 'avg_v_sur'] = target_period['v_sur'].mean()
        no_hard_braking = (target_period['acc_ego'].min()>-1.5)

        # Determine conflict and warning
        if no_hard_braking:
            records.loc[event_id, 'safety_recorded'] = True
            records.loc[event_id, 'safety_period'] = len(target_period)/10
            target_period = determine_conflicts(target_period, indicator, threshold)
            if np.any(target_period['conflict']):
                records.loc[event_id, 'false warning'] = 1
            else:
                records.loc[event_id, 'false warning'] = 0
    records['threshold'] = threshold
    return records


def optimize_threshold(warning, conflict_indicator, return_stats=False):
    danger = warning[warning['danger_recorded']].copy()
    tpr = danger.groupby('threshold')['true warning'].sum()/danger.groupby('threshold')['true warning'].size()
    tpr = tpr.reset_index().rename(columns={'true warning':'true positive rate'})
    safety = warning[warning['safety_recorded']].copy()
    fpr = safety.groupby('threshold')['false warning'].sum()/safety.groupby('threshold')['false warning'].size()
    fpr = fpr.reset_index().rename(columns={'false warning':'false positive rate'})
    statistics = tpr.reset_index().merge(fpr.reset_index(), on='threshold', how='outer')[['threshold','true positive rate','false positive rate']]
    if conflict_indicator=='TTC' or conflict_indicator=='MTTC':
        statistics = statistics.sort_values(by=['false positive rate','true positive rate','threshold'])
    else:
        statistics = statistics.sort_values(by=['false positive rate','true positive rate','threshold'], ascending=[True, True, False])

    statistics['combined rate'] = (1-statistics['true positive rate'])**2+(statistics['false positive rate'])**2
    optimal_rate = statistics['combined rate'].min()
    optimal_warning = statistics.loc[statistics['combined rate']==optimal_rate, ['threshold','true positive rate','false positive rate']]
    optimal_threshold = optimal_warning.iloc[0]
    print(conflict_indicator, 
            ' optimal threshold: ', optimal_threshold['threshold'], 
            ' true positive rate: ', round(optimal_threshold['true positive rate']*100, 2),
            ' false positive rate: ', round(optimal_threshold['false positive rate']*100, 2))
    if return_stats:
        optimal_warning = warning[warning['threshold']==optimal_threshold['threshold']]
        return statistics, optimal_warning, optimal_threshold
    else:
        return optimal_threshold['threshold']


def issue_warning(indicator, threshold, safety_evaluation, event_data, event_meta):
    event_data = event_data.reset_index().set_index(['event_id', 'target_id', 'time'])
    safety_evaluation = safety_evaluation.sort_values(['target_id','time'])
    events = safety_evaluation.set_index('event_id')
    event_ids = np.intersect1d(event_meta.index.values, events.index.unique())

    records = event_meta[['danger_start', 'danger_end', 'reaction_timestamp']].copy()
    for event_id in event_ids:
        event = events.loc[event_id].copy()
        danger = event[(event['time']>=event_meta.loc[event_id, 'danger_start']/1000)&
                       (event['time']<=event_meta.loc[event_id, 'danger_end']/1000)].reset_index()
        if len(danger)<5:
            records.loc[event_id, 'danger_recorded'] = False
            continue

        # Determine the conflicting target
        target_id = determine_target(indicator, danger)
        records.loc[event_id, 'target_id'] = target_id
        if np.isnan(target_id):
            records.loc[event_id, 'danger_recorded'] = False
            continue
        target = event[event['target_id']==target_id]

        # Determine first warning moment: the last safe->unsafe transition moment before impact_timestamp
        target = determine_conflicts(target, indicator, threshold)

        warning = target[target['time']<=event_meta.loc[event_id, 'impact_timestamp']/1000]['conflict'].astype(int).values
        warning_change = warning[1:] - warning[:-1]
        first_warning = np.where(warning_change==1)[0]
        if len(first_warning)>0:
            records.loc[event_id,'first_warning_timestamp'] = int(target.iloc[first_warning[-1]+1]['time']*1000)
        else:
            records.loc[event_id,'first_warning_timestamp'] = np.nan

    records['indicator'] = indicator
    records['optimal_threshold'] = threshold
    return records
