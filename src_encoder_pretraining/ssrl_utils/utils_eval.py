'''
This script includes utility functions for evaluation of the pretrained encoders.
'''

import os
import sys
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssrl_utils.utils_distance_matrix import get_EUC
from modules.measures import MeasureCalculator


class Multi_Evaluation:
    def __init__(self, data, latent):
        self.data = data
        self.latent = latent

    def define_ks(self, dist_mat_X):
        # define k values for evaluation, logarithmically spaced
        k_neighbours = np.unique(np.logspace(1, np.log(min(dist_mat_X.shape[0]/3,200))/np.log(5), num=10, base=5).astype(int))
        return k_neighbours

    def get_multi_evals(self, local=False):
        """
        Performs multiple evaluations for nonlinear dimensionality
        reduction.

        - data: data samples as matrix
        - latent: latent samples as matrix
        - local: whether to use local or global evaluation
        - ks: list of k values for evaluation
        """
        if local:
            dep_measures_list = {'mean_shared_neighbours':0., 
                                 'mean_dist_mrre':0., 
                                 'mean_trustworthiness':0, 
                                 'mean_continuity':0.}
            
            N = self.data.shape[1] # time series length
            sample_indices = np.arange(self.data.shape[0])
            sample_count = 0
            dist_mat_measure = {'local_distmat_rmse': 0}
            for sample_index in tqdm(sample_indices, desc='Local evaluation', ascii=True, miniters=100):
                data = self.data[sample_index].reshape(N, -1)
                latent = self.latent[sample_index].reshape(N, -1)
                dist_mat_X = get_EUC(data)
                dist_mat_Z = get_EUC(latent)
                if dist_mat_X.max()-dist_mat_X.min() == 0 or dist_mat_Z.max()-dist_mat_Z.min() == 0:
                    continue
                else:
                    dist_mat_X = abs((dist_mat_X - dist_mat_X.min()) / (dist_mat_X.max() - dist_mat_X.min()))
                    dist_mat_Z = abs((dist_mat_Z - dist_mat_Z.min()) / (dist_mat_Z.max() - dist_mat_Z.min()))

                dist_mat_measure['local_distmat_rmse'] += np.sqrt(np.mean((dist_mat_X - dist_mat_Z)**2))

                ks = self.define_ks(dist_mat_X)
                calc = MeasureCalculator(dist_mat_X, dist_mat_Z, max(ks))

                dep_measures = calc.compute_measures_for_ks(ks)
                mean_dep_measures = {'mean_'+key: np.nanmean(values) for key, values in dep_measures.items()}
                for key, value in mean_dep_measures.items():
                    dep_measures_list[key] += value

                sample_count += 1
                if sample_count >= 500:
                    break
            
            dist_mat_measure['local_distmat_rmse'] /= sample_count
            dep_measures = {'local_'+key: value/sample_count for key, value in dep_measures_list.items()}
            results = {**dist_mat_measure, **dep_measures}
        else:
            N = self.data.shape[0]
            print('Calculating global distance matrix...')
            dist_mat_X = get_EUC(self.data.reshape(N, -1))
            dist_mat_Z = get_EUC(self.latent.reshape(N, -1))
            dist_mat_X = abs((dist_mat_X - dist_mat_X.min()) / (dist_mat_X.max() - dist_mat_X.min()))
            dist_mat_Z = abs((dist_mat_Z - dist_mat_Z.min()) / (dist_mat_Z.max() - dist_mat_Z.min()))
            print('Distance matrix calculated.')

            dist_mat_measure = {'global_distmat_rmse': np.sqrt(np.mean((dist_mat_X - dist_mat_Z)**2))}

            ks = self.define_ks(dist_mat_X)
            calc = MeasureCalculator(dist_mat_X, dist_mat_Z, max(ks))

            dep_measures = calc.compute_measures_for_ks(ks)
            mean_dep_measures = {'global_mean_' + key: np.nanmean(values) for key, values in dep_measures.items()}

            results = {**dist_mat_measure, **mean_dep_measures}
            
        return results


def evaluate(data, model, batch_size, local=False):
    # encode data into latent space
    if local:
        latent = model.encode(data, batch_size=batch_size).detach().cpu().numpy() # (N, T, P)
    else:
        latent = model.encode(data, batch_size=batch_size, encoding_window='full_series').detach().cpu().numpy() # (N, P)

    evaluator = Multi_Evaluation(data, latent)
    ev_result = evaluator.get_multi_evals(local)

    return ev_result

