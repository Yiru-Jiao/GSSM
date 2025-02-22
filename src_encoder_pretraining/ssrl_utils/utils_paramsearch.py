'''
This script defines the grid search utilities for hyperparameter tuning of profile encoder pretraining.
'''

import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssrl_utils.utils_general import *
import ssrl_utils.utils_data as datautils
from clt_model import spclt


manual_seed = 131

# Define trainer
class trainer():
    def __init__(self, dist_metric='DTW', tau_inst=0, tau_temp=0, temporal_hierarchy=None, 
                 bandwidth=1., batch_size=8, weight_lr=0.05):
        self.dist_metric = dist_metric
        self.tau_inst = tau_inst
        self.tau_temp = tau_temp
        self.temporal_hierarchy = temporal_hierarchy
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self.weight_lr = weight_lr

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def define_encoder(self, sim_mat, input_dims, device, regularizer=None):
        self.model_config = dict(
            input_dims = input_dims,
            output_dims = 256,
            dist_metric = self.dist_metric,
            device = device,
            batch_size = self.batch_size,
            lr = 0.001,
            weight_lr = self.weight_lr,
            loss_config = dict(
                tau_inst = self.tau_inst,
                tau_temp = self.tau_temp,
                lambda_inst = 0.5,
                temporal_hierarchy = self.temporal_hierarchy),
            regularizer_config = dict(
                reserve = regularizer,
                bandwidth = self.bandwidth),
            )
        self.encoder = spclt(**self.model_config)

        self.soft_assignments = datautils.assign_soft_labels(sim_mat, self.tau_inst)
        if self.soft_assignments is None:
            print('Soft assignment is not used in this run.')
        
        return self

    def fit(self, train_data, sim_mat, dataset, encoder_config=None):
        encoder_config['input_dims'] = train_data.shape[-1]
        self = self.define_encoder(sim_mat, **encoder_config)
        self.loss_log = self.encoder.fit(dataset, train_data, self.soft_assignments, 
                                         scheduler='constant', verbose=2)
        return self
    
    def get_params(self, deep=False):
        return dict(
            tau_inst = self.tau_inst,
            tau_temp = self.tau_temp,
            temporal_hierarchy = self.temporal_hierarchy,
            bandwidth = self.bandwidth,
            batch_size = self.batch_size,
            weight_lr = self.weight_lr
        )

    def score(self, test_data):
        soft_assignments = datautils.assign_soft_labels(None, self.tau_inst)
        return -self.encoder.compute_loss(test_data, soft_assignments)
    

def grid_search(params, dataset, dist_metric, 
                train_data, sim_mat, n_fold, n_jobs, fit_config):
    if n_fold == 0:
        n_fold = ShuffleSplit(n_splits=1, test_size=0.3, random_state=manual_seed)

    scorer = trainer(dist_metric=dist_metric)
    gs = GridSearchCV(scorer, params, cv=n_fold, n_jobs=n_jobs, verbose=0, refit=False)
    gs.fit(train_data, **{'sim_mat':sim_mat, 'dataset': dataset, 'encoder_config': fit_config})
    best_params, best_score = gs.best_params_, round(gs.best_score_, 4)

    del train_data
    del scorer
    del gs
    torch.cuda.empty_cache()
    return best_params, best_score
