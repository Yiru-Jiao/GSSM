'''
This script searches the hyperparameters for the contrastive learning training of the profile encoder.

Search strategy:
Fix `lr`=0.001 and `bs`=8 (as learned in our previous research), 
the training score is the contrastive learning loss (without regularization)

- SoftCLT (use soft labels, no regularizer):
  Phase 1: with other parameters default, search for best `tau_temp` and `temporal_hierarchy`
  Phase 2: with best `tau_temp` and `temporal_hierarchy`, search for best `tau_inst` and `batch_size`

- TopoSoftCLT (use soft labels, topology regularizer):
  Phase 0: set default `tau_inst`, `tau_temp`, `temporal_hierarchy`, `batch_size` to the best tuned values from SoftCLT
  Phase 1: with other parameters default, search for best `weight_lr`

- GGeoSoftCLT (use soft labels, geometry regularizer):
  Phase 0: set default `tau_inst`, `tau_temp`, `temporal_hierarchy`, `batch_size` to the best tuned values from SoftCLT
  Phase 1: with other parameters default, search for best `bandwidth` and `weight_lr`

-------------------------------------------------------------------------------------------------------
|            |  SoftCLT   | TopoSoftCLT | GGeoSoftCLT |  in total  |
|    runs    |  5x3+5x1   |      3      |     5x3     |     38     |
-------------------------------------------------------------------------------------------------------
'''

import os
import sys
import time as systime
import pandas as pd
import argparse
from ssrl_utils.utils_paramsearch import *
import ssrl_utils.utils_data as datautils

manual_seed = 131 
path_prepared = './PreparedData/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--n_fold', type=int, default=0, help='The number of folds for cross-validation (defaults to 0 for no cross-validation)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='The number of parallel jobs to run for grid search (defaults to -1 for all available cores)')
    args = parser.parse_args()
    return args


def main(args):
    initial_time = systime.time()
    fix_seed(manual_seed, deterministic=True) # Fix the random seed for consistent search
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    
    # Initialize the deep learning program, `init_dl_program` is defined in `utils_general.py`
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__}, Available threads: {os.cpu_count()} ---')
    
    def use_best_params(best_param_log, phase):
        best_params = best_param_log[phase]
        if 'score' in best_params:
            best_params.pop('score')
        for key, value in best_params.items():
            if key == 'batch_size':
                best_params[key] = [int(value)]
            elif key == 'temporal_hierarchy':
                if value == 'linear' or value == 'exponential':
                    best_params[key] = [value]
                else:
                    best_params[key] = [None]
            else:
                best_params[key] = [value]
        return best_params

    def search_best_params(parameters2search, params, search_space, grid_search_args):
        for parameter in parameters2search:
            params = {**params, parameter: search_space[parameter]}
        best_params, best_score = grid_search(params, **grid_search_args)
        for parameter in parameters2search:
            params = {**params, parameter: [best_params[parameter]]}
        return params, best_score
    
    def save_best_params(best_param_log, log_dir):
        log2save = best_param_log.copy()
        for tuned_phase, tuned_params in log2save.items():
            log2save[tuned_phase] = {key:value[0] if isinstance(value, list) else value for key,value in tuned_params.items()}
        log2save = pd.DataFrame(log2save).T # index: phase, columns: hyperparameters and score
        log2save.to_csv(log_dir)

    start_time = systime.time()
    # Load dataset
    for dataset in ['SafeBaseline','INTERACTION_highD_Argoverse_SafeBaseline']:
        print(f'---- Loading {dataset} data ----')
        if '_' in dataset:
            train_data, _ = datautils.load_data(dataset.split('_'), dataset_dir=path_prepared, feature='profiles')
        else:
            train_data, _ = datautils.load_data([dataset], dataset_dir=path_prepared, feature='profiles')
    
        dist_metric = 'DTW'
        sim_mat = None # to be computed per batch during training
        
        # Set result-saving directory
        save_dir = f'{path_prepared}EncoderPretraining/spclt/{dataset}/'
        os.makedirs(save_dir, exist_ok=True)

        # Predefine default params and search spacef
        default_params = {'tau_inst': [0],
                          'tau_temp': [0],
                          'temporal_hierarchy': [None],
                          'bandwidth': [1.],
                          'batch_size': [8],
                          'weight_lr': [0.05]}

        # Define the search space
        search_space = {'tau_inst': [1, 3, 5, 10, 20], # used in softclt study
                        'tau_temp': [0.5, 1., 1.5, 2., 2.5], # used in softclt study
                        'temporal_hierarchy': [None, 'linear', 'exponential'],
                        'bandwidth': [0.25, 1., 9., 25., 49.], # used in geometry regularizer only
                        'batch_size': [8],
                        'weight_lr': [0.005, 0.01, 0.05]}
        print(f"--- batch_size search space: {search_space['batch_size']} ---")

        # Initialize the best_param_log
        log_dir = os.path.join(save_dir, f'representation_hyperparameters.csv')
        if os.path.exists(log_dir):
            best_param_log = pd.read_csv(log_dir, index_col=0)
            best_param_log = best_param_log.to_dict(orient='index')
        else:
            best_param_log = {}

        # Define the grid search arguments
        grid_search_args = {'dataset': dataset,
                            'dist_metric': dist_metric,
                            'sim_mat': sim_mat,
                            'train_data': train_data,
                            'n_fold': args.n_fold,
                            'n_jobs': args.n_jobs,
                            'fit_config': {'device': device, 'regularizer': None}}

        # Initialize the dict of parameters
        params = default_params.copy()

        # SoftCLT (use soft labels, no regularizer)
        grid_search_args['fit_config'] = {'device': device, 'regularizer': None}
        
        if 'SoftCLT_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'SoftCLT_Phase1')
            print(f'--- SoftCLT_Phase1 search already completed ---')
        else:
            params, best_score = search_best_params(['tau_temp', 'temporal_hierarchy'], params, search_space, grid_search_args)
            best_param_log['SoftCLT_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- SoftCLT_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        if 'SoftCLT_Phase2' in best_param_log:
            params = use_best_params(best_param_log, 'SoftCLT_Phase2')
            print(f'--- SoftCLT_Phase2 search already completed ---')
        else:
            params, best_score = search_best_params(['tau_inst', 'batch_size'], params, search_space, grid_search_args)
            best_param_log['SoftCLT_Phase2'] = params
            save_best_params(best_param_log, log_dir)
            print('--- SoftCLT_Phase2 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        # TopoSoftCLT (use soft labels, topology regularizer)
        grid_search_args['fit_config'] = {'device': device, 'regularizer': 'topology'}
        
        if 'TopoSoftCLT_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'TopoSoftCLT_Phase1')
            print(f'--- TopoSoftCLT_Phase1 search already completed ---')
        else:
            params, best_score = search_best_params(['weight_lr'], params, search_space, grid_search_args)
            best_param_log['TopoSoftCLT_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- TopoSoftCLT_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        # GGeoSoftCLT (use soft labels, geometry regularizer)
        grid_search_args['fit_config'] = {'device': device, 'regularizer': 'geometry'}

        if 'GGeoSoftCLT_Phase1' in best_param_log:
            params = use_best_params(best_param_log, 'GGeoSoftCLT_Phase1')
            print(f'--- GGeoSoftCLT_Phase1 hyperparameter search already completed ---')
        else:
            params, best_score = search_best_params(['bandwidth', 'weight_lr'], params, search_space, grid_search_args)
            best_param_log['GGeoSoftCLT_Phase1'] = params
            save_best_params(best_param_log, log_dir)
            print('--- GGeoSoftCLT_Phase1 | time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + f' | best score: {best_score} ---')

        print(f'--- {dataset} hyperparameter search completed, time elapsed : ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time()-start_time)) + ' ---')
        
    print('--- Time elapsed in total : ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time()-initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
    