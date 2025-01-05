'''
'''

import os
import sys
import torch
import numpy as np
from scipy.special import erf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test


def define_model(device, path_prepared, encoder_selection, pretrained_encoder, batch_size, initial_lr):
    # Define the model
    pipeline = train_val_test(device, path_prepared, encoder_selection, pretrained_encoder)
    ## Load trained model
    pipeline.load_model(batch_size, initial_lr)
    print('Posterior inference model loaded.')
    return pipeline.model


def lognormal_cdf(x, mu, sigma):
    return 1/2+1/2*erf((np.log(x)-mu)/sigma/np.sqrt(2))


def extreme_cdf(x, mu, sigma, n=10):
    return (1-lognormal_cdf(x,mu,sigma))**n


def send_x_to_device(x, device):
    if isinstance(x, tuple):
        return [torch.from_numpy(i).float().to(device) for i in x]
    else:
        return torch.from_numpy(x).float().to(device)


def assess_conflict(states, model, device, output='probability', n=25):
    x, proximity = states

    # Compute mu and sigma
    with torch.no_grad():
        out = model(send_x_to_device(x, device))
        mu, sigma, a = out
    mu = mu.squeeze().cpu().numpy()
    sigma = sigma.squeeze().cpu().numpy()
    a = a.squeeze().cpu().numpy()

    probability = extreme_cdf(proximity, mu, sigma, n)
    # 0.5 means that the probability of conflict is larger than the probability of non-conflict
    max_intensity = np.log(0.5)/np.log(1-lognormal_cdf(proximity, mu, sigma)+1e-6)
    max_intensity = np.maximum(1., max_intensity)

    if output=='probability':
        return probability
    elif output=='intensity':
        return max_intensity
    elif output=='both':
        return probability, max_intensity
    elif output=='all':
        return mu, sigma, a, probability, max_intensity
