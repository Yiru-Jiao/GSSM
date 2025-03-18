'''
This script contains utility functions for attributing potential conflict intensity to features.
'''

import os
import sys
import shap
import random
import time as systime
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.special import erf
import torch
import argparse
from sklearn.preprocessing import OneHotEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()
from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test
from src_safety_evaluation.validation_utils.utils_evaluation import read_events, set_veh_dimensions


def create_categorical_encoder(events, environment_feature_names):
    categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    events.loc[events['surfaceCondition']=='Other','surfaceCondition'] = 'Unknown'
    data2fit = events[environment_feature_names].fillna('Unknown')
    data2fit = data2fit.loc[(data2fit!='Unknown').all(axis=1)]
    categorical_encoder.fit(data2fit.values)
    return categorical_encoder


class get_sample(): 
    def __init__(self, encoder_selection, path_result):
        event_categories = sorted(os.listdir(path_result + 'EventData/'))
        profiles_features = []
        current_features = []
        spacing_list = []
        event_id_list = []
        for event_cat in event_categories:
            event_featurs = np.load(path_result + f'EventData/{event_cat}/event_features.npz')
            profiles_features.append(event_featurs['profiles'])
            current_features.append(event_featurs['current'])
            spacing_list.append(event_featurs['spacing'])
            event_id_list.append(event_featurs['event_id'])
        profiles_features = np.concatenate(profiles_features, axis=0)
        current_features = np.concatenate(current_features, axis=0)
        self.spacing_list = np.concatenate(spacing_list, axis=0)
        event_id_list = np.concatenate(event_id_list, axis=0)
        self.num_samples = event_id_list.shape[0]

        # Define one-hot encoder for environment features
        if 'environment' in encoder_selection:
            events = pd.read_csv('./RawData/SHRP2/HondaDataSupport/InsightTables_csv/Event_Table.csv').set_index('eventID')
            environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
            one_hot_encoder = create_categorical_encoder(events, environment_feature_names)

        states = []
        variables = []
        if encoder_selection[0]=='current':
            states.append(current_features[:,list(range(11))+[-1]])
            variables.extend(['Ego veh length','Sur veh length','Combined width','Ego speed','Sur lat speed','Sur lon speed',
                              'Squared ego speed','Squared sur speed','Squared relative speed','Relative speed','Relative heading','2D spacing direction'])
        if encoder_selection[0]=='current+acc':
            states.append(current_features)
            variables.extend(['Ego veh length','Sur veh length','Combined width','Ego speed','Sur lat speed','Sur lon speed',
                              'Squared ego speed','Squared sur speed','Squared relative speed','Relative speed',
                              'Relative heading','Ego acceleration','2D spacing direction'])
        if 'environment' in encoder_selection:
            environment_features = events.loc[event_id_list[:,0], environment_feature_names].fillna('Unknown')
            environment_features = one_hot_encoder.transform(environment_features.values)
            states.append(environment_features)
            variables.extend(['Lighting', 'Weather', 'Road surface', 'Traffic density'])
        if 'profiles' in encoder_selection:
            states.append(profiles_features)
            variables.extend(['Passed 0.5s','Passed 1s','Passed 1.5s','Passed 2s','Passed 2.5s'])
        self.states = states
        self.variables = variables

        self.event_id_list = pd.DataFrame(event_id_list, columns=['event_id','target_id','time'])
        self.event_id_list['idx'] = np.arange(self.num_samples)
        self.event_id_list = self.event_id_list.set_index(['event_id','target_id','time'])
        if len(self.states)>1:
            def get_item(event_id, target_id, time=None):
                if time is None:
                    idx = self.event_id_list.loc[(event_id, target_id, slice(None))]['idx'].values[::-1]
                else:
                    idx = self.event_id_list.loc[(event_id, target_id, time)]['idx']
                    if isinstance(idx, np.int32):
                        idx = [idx]
                    else:
                        idx = idx.values
                samples = tuple([torch.from_numpy(x_i[idx]).float() for x_i in self.states])
                return samples, self.spacing_list[idx]
        else:
            def get_item(event_id, target_id, time=None):
                if time is None:
                    idx = self.event_id_list.loc[(event_id, target_id, slice(None))]['idx'].values[::-1]
                else:
                    idx = self.event_id_list.loc[(event_id, target_id, time)]['idx']
                    if isinstance(idx, np.int32):
                        idx = [idx]
                    else:
                        idx = idx.values
                return torch.from_numpy(self.states[0][idx]).float(), self.spacing_list[idx]
        self.get_item = get_item


