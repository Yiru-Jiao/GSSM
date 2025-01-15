'''
This script is used to pretrain and evaluate the profile encoder.
Tuned hyperparameters are loaded from locally saved files.
'''

import os
import sys
import time as systime
import glob
import torch
import random
import numpy as np
import pandas as pd
import argparse
from ssrl_utils.utils_paramsearch import *
import ssrl_utils.utils_data as datautils
from clt_model import spclt
from ssrl_utils.utils_general import *
from ssrl_utils.utils_eval import *

manual_seed = 131
path_prepared = './PreparedData/'
path_processed = './ProcessedData/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)

    # Set default parameters
    args.sliding_padding = 0
    args.repr_dims = 64
    args.dist_metric = 'DTW'
    args.tau_inst = 0
    args.tau_temp = 0
    args.temporal_hierarchy = None
    args.regularizer = None
    args.bandwidth = 1.
    args.iters = None
    args.epochs = 50
    args.batch_size = 8
    args.lr = 0.002
    args.weight_lr = 0.01

    return args


def main(args):
    initial_time = systime.time()
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    
    # Set the random seed
    if args.reproduction:
        args.seed = manual_seed # Fix the random seed for reproduction
    if args.seed is None:
        args.seed = random.randint(0, 1000)
    print(f"Random seed is set to {args.seed}")
    fix_seed(args.seed, deterministic=args.reproduction)
    
    # Initialize the deep learning program
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    # Create the directory to save the evaluation results
    run_dir = f'{path_prepared}EncoderPretraining/spclt/trained_models/'
    results_dir = f'{path_prepared}EncoderPretraining/spclt/evaluation.csv'
    os.makedirs(run_dir, exist_ok=True)

    # Define metrics
    knn_metrics = ['mean_shared_neighbours', 'mean_dist_mrre', 'mean_trustworthiness', 'mean_continuity'] # kNN-based, averaged over various k
    
    def read_saved_results():
        eval_results = pd.read_csv(results_dir)
        eval_results = eval_results.set_index('model')
        return eval_results
    
    model_list = ['ts2vec', 'topo-ts2vec', 'ggeo-ts2vec', 'softclt', 'topo-softclt', 'ggeo-softclt']
    if os.path.exists(results_dir):
        eval_results = read_saved_results()
    else:
        metrics = ['local_'+metric for metric in knn_metrics] + ['global_'+metric for metric in knn_metrics]
        eval_results = pd.DataFrame(np.zeros((len(model_list), 8), dtype=np.float32), columns=metrics, index=model_list)
        eval_results.index.name = 'model'
        eval_results.to_csv(results_dir)
    
    # Load dataset
    print('---- Loading data ----')
    train_data, _ = datautils.load_data(dataset_dir=path_prepared)
    dataset = 'SafeBaseline'
    train_sim_mat = None # to be computed per batch during training
    
    # Load tuned hyperparameters
    tuned_params_dir = f'{path_prepared}EncoderPretraining/spclt/representation_hyperparameters.csv'
    if os.path.exists(tuned_params_dir):
        tuned_params = pd.read_csv(tuned_params_dir, index_col=0)
    else:
        print(f'****** {tuned_params_dir} not found ******')
        sys.exit(0)

    # Iterate over different models
    feature_size = train_data.shape[-1]
    verbose = 12 # update per n_epochs // (1+verbose*4) epoch

    for model_type in model_list:
        if eval_results.loc[model_type, 'global_mean_continuity'] > 0:
            final_epoch = eval_results.loc[model_type, 'model_used'].split('epo')[0].split('_')[-1]
            print(f'--- {model_type} has been trained until epoch {final_epoch}, skipping evaluation ---')
            continue
        start_time = systime.time()
        save_dir = os.path.join(run_dir, f'{model_type}')
        os.makedirs(save_dir, exist_ok=True)

        # Set hyperparameters and configure model
        args = load_tuned_hyperparameters(args, tuned_params, model_type)
        model_config = configure_model(args, feature_size, device)

        # Train model if not already trained
        if os.path.exists(f'{save_dir}/loss_log.csv'):
            print(f'--- {model_type} has been trained, loading final model ---')
        else:
            # Create model
            model_config['after_epoch_callback'] = save_checkpoint_callback(save_dir, 0, unit='epoch')
            model = spclt(**model_config)

            scheduler = 'reduced'
            print(f'--- {model_type} training with ReduceLROnPlateau scheduler ---')
            soft_assignments = datautils.assign_soft_labels(train_sim_mat, args.tau_inst)
            loss_log = model.fit(dataset, train_data, soft_assignments, args.epochs, args.iters, scheduler, verbose=verbose)
            # Save loss log
            save_loss_log(loss_log, save_dir, regularizer=args.regularizer)
            print(f'Training time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)))
        
        # Reserve the latest model and remove the rest
        existing_models = glob.glob(f'{save_dir}/*_net.pth')
        if len(existing_models)>1:
            existing_models.sort(key=os.path.getmtime, reverse=True)
            for model_epoch in existing_models[1:]:
                os.remove(model_epoch)
                if model_type in ['topo-ts2vec', 'topo-softclt']:
                    os.remove(model_epoch.replace('_net.pth', '_loss_log_vars.npy'))
        best_model = 'model' + existing_models[0].split('model')[-1].split('_net')[0]

        # Load best model for evaluation
        model = spclt(**model_config)
        model.load(f'{save_dir}/{best_model}')

        # Evaluate model for UEA datasets
        print(f'Evaluating with {best_model} ...')
        ## reload data for evaluation
        loaded_data = datautils.load_data(dataset_dir=path_prepared)
        _, test_data = loaded_data
        
        ## distance and density results
        local_dist_dens_results = evaluate(test_data, model, batch_size=128, local=True)
        global_dist_dens_results = evaluate(test_data, model, batch_size=128, local=False)

        ## loss results
        test_sim_mat = None
        test_soft_assignments = datautils.assign_soft_labels(test_sim_mat, args.tau_inst)
        loss_results = model.compute_loss(test_data, test_soft_assignments, non_regularized=False)
        loss_results = {'cl_loss': loss_results[1],
                        'sp_loss': loss_results[3] if args.regularizer is not None else np.nan}

        # Save evaluation results
        key_values = {**loss_results, **local_dist_dens_results, **global_dist_dens_results}
        keys = list(key_values.keys())
        values = np.array(list(key_values.values())).astype(np.float32)
        eval_results = read_saved_results() # read saved results again to avoid overwriting
        eval_results.loc[model_type, keys] = values
        eval_results.loc[model_type, 'model_used'] = best_model

        # Save evaluation results per dataset and model
        eval_results.to_csv(results_dir)
            
    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
