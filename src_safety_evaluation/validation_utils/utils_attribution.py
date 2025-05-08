'''
This script contains utility functions for attributing potential conflict intensity to features.
'''

import os
import sys
import numpy as np
import pandas as pd
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()


class get_sample():
    '''
    This class is used to get the sample data for a specific event and target vehicle.
    '''
    def __init__(self, encoder_selection, path_result):
        event_categories = sorted(os.listdir(path_result + 'EventData/'))
        profiles_features = []
        current_features = []
        environment_features = []
        spacing_list = []
        event_id_list = []
        for event_cat in event_categories:
            event_featurs = np.load(path_result + f'EventData/{event_cat}/event_features.npz')
            profiles_features.append(event_featurs['profiles'])
            current_features.append(event_featurs['current'])
            environment_features.append(event_featurs['environment'])
            spacing_list.append(event_featurs['spacing'])
            event_id_list.append(event_featurs['event_id'])
        profiles_features = np.concatenate(profiles_features, axis=0)
        current_features = np.concatenate(current_features, axis=0)
        environment_features = np.concatenate(environment_features, axis=0)
        self.spacing_list = np.concatenate(spacing_list, axis=0)
        event_id_list = np.concatenate(event_id_list, axis=0)
        self.num_samples = event_id_list.shape[0]

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


