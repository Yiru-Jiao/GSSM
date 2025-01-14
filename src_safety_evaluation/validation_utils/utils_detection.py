'''
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


def read_events(path_events):
    event_categories = sorted(os.listdir(path_events))
    event_meta = pd.concat([pd.read_csv(path_events + f'{event_cat}/event_meta.csv') for event_cat in event_categories])
    event_meta = event_meta.set_index('event_id')
    event_data = pd.concat([pd.read_hdf(path_events + f'{event_cat}/event_data.h5', key='data') for event_cat in event_categories])
    assert np.all(np.isin(event_data['event_id'].unique(), event_meta.index.values))
    return event_meta, event_data


def read_evaluation(pretraining, encoder_name, cross_attention_name, path_events):
    event_categories = sorted(os.listdir(path_events))
    safety_evaluation = pd.concat([pd.read_hdf(path_events + f'{event_cat}/{pretraining}/{encoder_name}_{cross_attention_name}.h5', key='data') for event_cat in event_categories])
    return safety_evaluation


def define_model(device, path_prepared, encoder_selection, cross_attention, pretrained_encoder):
    # Define the model
    pipeline = train_val_test(device, path_prepared, encoder_selection, cross_attention, pretrained_encoder, return_attention=True)
    ## Load trained model
    pipeline.load_model()
    print(f'Model loaded: {pipeline.encoder_name}-{pipeline.cross_attention_name}')
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
    

def SSSE(states, model, device):
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

    # 0.5 means that the probability of conflict is larger than the probability of non-conflict
    max_intensity = np.log(0.5)/np.log(1-lognormal_cdf(proximity, mu, sigma)+1e-6)
    max_intensity = np.maximum(1., max_intensity)

    return mu, sigma, max_intensity


def determine_conflicts(evaluation, conflict_indicator, threshold):
    evaluation = evaluation.reset_index()
    evaluation['conflict'] = False

    if conflict_indicator=='TTC':
        evaluation.loc[(evaluation['TTC']<threshold), 'conflict'] = True
        return evaluation
    
    elif conflict_indicator=='DRAC':
        evaluation.loc[(evaluation['DRAC']>threshold), 'conflict'] = True
        return evaluation
    
    elif conflict_indicator=='SSSE':
        evaluation['probability'] = extreme_cdf(evaluation['proximity'].values, evaluation['mu'].values, evaluation['sigma'].values, threshold)
        # 0.5 means that the probability of conflict is larger than the probability of non-conflict
        evaluation.loc[evaluation['probability']>0.5, 'conflict'] = True
        return evaluation


def roc_curve(evaluation, event_meta, indicator, thresholds):
    evaluation = evaluation.sort_values(['target_id','time'])
    events = events.set_index('event_id')
    event_meta = event_meta[event_meta['reaction_covered']]
    event_ids = event_meta.index.values

    for event_id in event_ids:
        event = events.loc[event_id].copy()
        # the 3 seconds in the event before the moment are supposed to be dangerous
        within_3s = data[(data['time']>=moment-3)&(data['time']<=moment)&(data['event'])]
        true_warning = within_3s[within_3s['conflict']]
        if len(true_warning)>0:
            meta.loc[trip_id,'true warning'] = True
        else:
            meta.loc[trip_id,'true warning'] = False

        # the first 3 seconds are assumed to be safe
        beginning = data['time'].min()
        first_3s = data[(data['time']>=beginning)&(data['time']<=beginning+3)&(~data['event'])]
        false_warning = first_3s[first_3s['conflict']]
        if len(false_warning)>0:
            meta.loc[trip_id,'false warning'] = True
        else:
            meta.loc[trip_id,'false warning'] = False

    return results


def issue_warning(evaluation, event_meta, indicator, threshold):
    evaluation = evaluation.sort_values(['target_id','time'])
    events = events.set_index('event_id')
    event_meta = event_meta[event_meta['reaction_covered']]
    event_ids = event_meta.index.values

    for event_id in event_ids:
        event = events.loc[event_id].copy()

        data = determine_conflicts(data, indicator, parameters)
        true_warning = data[(data['conflict'])&(data['event'])]
        event_period = data[data['event']]
        meta.loc[trip_id,'warning period'] = len(true_warning)/len(event_period)

        # record the first warning before the impact moment
        indicated_events.append(data)
        warning = data[data['time']<=moment]['conflict'].astype(int).values
        warning_change = warning[1:] - warning[:-1]
        first_warning = np.where(warning_change==1)[0]
        if len(first_warning)>0:
            meta.loc[trip_id,'first warning'] = data.loc[first_warning[-1]+1,'time']
        else:
            meta.loc[trip_id,'first warning'] = np.nan
        
    return meta.reset_index()