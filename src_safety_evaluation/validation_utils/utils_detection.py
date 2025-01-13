'''
'''

import os
import sys
import torch
import numpy as np
from scipy.special import erf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test


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
    if isinstance(x, tuple):
        return [torch.from_numpy(i).float().to(device) for i in x]
    else:
        return torch.from_numpy(x).float().to(device)


def SSSE(states, model, device):
    x, proximity = states

    # Compute mu and sigma
    with torch.no_grad():
        out = model(send_x_to_device(x, device))
        mu, sigma, _ = out
    mu = mu.squeeze().cpu().numpy()
    sigma = sigma.squeeze().cpu().numpy()

    # 0.5 means that the probability of conflict is larger than the probability of non-conflict
    max_intensity = np.log(0.5)/np.log(1-lognormal_cdf(proximity, mu, sigma)+1e-6)
    max_intensity = np.maximum(1., max_intensity)

    return mu, sigma, max_intensity


def determine_conflicts(data, conflict_indicator, parameters):
    data = data.reset_index()

    if conflict_indicator=='TTC':
        ttc_threshold = parameters[0]
        data['conflict'] = False
        data['indicator_value'] = TwoDimTTC.TTC(data, 'values')
        data.loc[(data['indicator_value']<ttc_threshold), 'conflict'] = True
        return data
    
    elif conflict_indicator=='DRAC':
        drac_threshold = parameters[0]
        data['s_box'] = TwoDimTTC.CurrentD(data, 'values')
        data.loc[data['s_box']<1e-6, 's_box'] = 1e-6
        data['conflict'] = False
        data['delta_v'] = np.sqrt((data['vx_i']-data['vx_j'])**2 + (data['vy_i']-data['vy_j'])**2)
        follower_speed = data['forward'].astype(int)*data['speed_i'] + (1-data['forward'])*data['speed_j']
        leader_speed = data['forward'].astype(int)*data['speed_j'] + (1-data['forward'])*data['speed_i']
        data['indicator_value'] = data['delta_v']**2 / 2 / data['s_box']
        data.loc[follower_speed<=leader_speed, 'indicator_value'] = 0.
        data.loc[(data['indicator_value']>drac_threshold), 'conflict'] = True
        return data
    
    elif conflict_indicator=='Unified':
        n, proximity_phi = parameters
        data['s_centroid'] = np.sqrt((data['x_i']-data['x_j'])**2 + (data['y_i']-data['y_j'])**2)
        data = data.merge(proximity_phi, on=['trip_id','time'])
        data['probability'] = extreme_cdf(data['s_centroid'].values, data['mu'].values, data['sigma'].values, n)
        data['conflict'] = False
        # 0.5 means that the probability of conflict is larger than the probability of non-conflict
        data.loc[data['probability']>0.5, 'conflict'] = True
        return data


def warning(events, meta, parameters, indicator, record_data=False):
    '''
    Perform warning analysis on the given events and metadata.

    Parameters:
    - events: Event data.
    - meta: Metadata.
    - parameters: Parameters for conflict detection.
    - indicator: Indicator for conflict detection ('TTC', 'PSD', or 'Unified').
    - record_data: Whether to record additional data.

    Returns:
    - Warning analysis results.
    '''
    events = events.sort_values(['trip_id','time'])
    events = events.set_index('trip_id')
    meta = meta.rename(columns={'webfileid':'trip_id'}).set_index('trip_id')
    meta = meta[['event start time','event end time','moment']].copy()
    trip_ids = meta.index

    ## Apply to each trip
    if record_data:
        indicated_events = []
    for trip_id in trip_ids:
        data = events.loc[trip_id].copy()
        moment = meta.loc[trip_id]['moment'] # moment of the minimum distance

        data = determine_conflicts(data, indicator, parameters)
        true_warning = data[(data['conflict'])&(data['event'])]
        event_period = data[data['event']]
        meta.loc[trip_id,'warning period'] = len(true_warning)/len(event_period)

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

        # record the first warning before the moment
        if record_data:
            indicated_events.append(data)
            warning = data[data['time']<=moment]['conflict'].astype(int).values
            warning_change = warning[1:] - warning[:-1]
            first_warning = np.where(warning_change==1)[0]
            if len(first_warning)>0:
                meta.loc[trip_id,'first warning'] = data.loc[first_warning[-1]+1,'time']
            else:
                meta.loc[trip_id,'first warning'] = np.nan
        
    if record_data:
        indicated_events = pd.concat(indicated_events).reset_index(drop=True)
        return meta.reset_index(), indicated_events
    else:
        return meta.reset_index()