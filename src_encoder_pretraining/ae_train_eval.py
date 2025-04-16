'''
This script pretrains and evaluates the encoders for current and environment features.
'''

import os
import sys
import time as systime
import glob
import numpy as np
import pandas as pd
import argparse
import torch
import random
from ae_model import autoencoder
import ssrl_utils.utils_data as datautils
from ssrl_utils.utils_general import *
from ssrl_utils.utils_eval import *

manual_seed = 131
path_prepared = './PreparedData/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_name', type=str, default='current', help='The encoder name to use for training and inference (defaults to current)')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    parser.add_argument('--reversed_list', type=int, default=0, help='Whether to reverse the datasets list (defaults to False), useful for running parallel jobs')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    args.reversed_list = bool(args.reversed_list)
    args.epochs = 1000
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

    # Define metrics
    knn_metrics = ['mean_shared_neighbours', 'mean_dist_mrre', 'mean_trustworthiness', 'mean_continuity'] # kNN-based, averaged over various k
    
    if 'current' in args.encoder_name:
        bslr_list = ['bs32_lr0.0001', 'bs64_lr0.0001', 'bs128_lr0.0001', 'bs256_lr0.0001']
    else:
        bslr_list = ['bs16_lr0.001', 'bs32_lr0.001', 'bs64_lr0.001', 'bs128_lr0.001']
    
    # Iterate over different datasets and hyperparameters
    verbose = 5 # update per n_epochs // (1+verbose*4) epoch
    if args.encoder_name == 'environment':
        datasets_list = [['SafeBaseline']]
    else:
        datasets_list = [['SafeBaseline', 'ArgoverseHV', 'highD']]

    if args.reversed_list:
        datasets_list = datasets_list[::-1]
        bslr_list = bslr_list[::-1]
    
    for datasets in datasets_list:
        # Create the directory to save the evaluation results
        dataset_name = '_'.join(datasets)
        run_dir = f'{path_prepared}EncoderPretraining/{args.encoder_name}_autoencoder/{dataset_name}/trained_models/'
        results_dir = f'{path_prepared}EncoderPretraining/{args.encoder_name}_autoencoder/{dataset_name}/evaluation.csv'
        os.makedirs(run_dir, exist_ok=True)

        if os.path.exists(results_dir):
            eval_results = pd.read_csv(results_dir)
        else:
            metrics = ['global_'+metric for metric in knn_metrics]
            eval_results = pd.DataFrame(np.zeros((len(bslr_list), 4), dtype=np.float32), columns=metrics)
            eval_results['bslr'] = bslr_list
            eval_results.to_csv(results_dir, index=False)
        
        print(f'---- Training and evaluating with datasets: {datasets} ----')
        train_data, test_data = datautils.load_data(datasets, dataset_dir=path_prepared, feature=args.encoder_name)
        if 'current' in args.encoder_name:
            if test_data.shape[0]>10000:
                test_data = test_data[np.random.choice(test_data.shape[0], 10000, replace=False)] # reduce test data size to avoid memory error

        start_time = systime.time()
        for bslr in bslr_list:
            args.batch_size = int(bslr.split('_')[0].replace('bs',''))
            args.lr = float(bslr.split('_')[1].replace('lr',''))

            save_dir = os.path.join(run_dir, f'{bslr}')
            os.makedirs(save_dir, exist_ok=True)

            if eval_results[eval_results['bslr']==bslr]['global_mean_shared_neighbours'].values[0]>0:
                final_epoch = eval_results[eval_results['bslr']==bslr]['model_used'].values[0].split('epo')[0].split('_')[-1]
                print(f'--- {bslr} has been trained until epoch {final_epoch}, skipping evaluation ---')
                continue

            # Train model if not already trained
            if os.path.exists(f'{save_dir}/loss_log.csv'):
                print(f'--- {bslr} has been trained, loading final model ---')
            else:
                # Create model
                model = autoencoder(args.encoder_name, args.lr, device, args.batch_size,
                                    after_epoch_callback = save_checkpoint_callback(save_dir, 0, unit='epoch'))
                scheduler = 'reduced'
                print(f'--- {bslr} training with ReduceLROnPlateau scheduler ---')
                train_losses, val_losses = model.fit(train_data, args.epochs, scheduler, verbose=verbose)
                # Save loss log
                loss_log = pd.DataFrame(index=[f'epoch_{i}' for i in range(1, len(train_losses)+1)],
                                        data={'train_loss': train_losses, 'val_loss': val_losses})
                loss_log.to_csv(f'{save_dir}/loss_log.csv')
            
            # Reserve the latest model and remove the rest
            existing_models = glob.glob(f'{save_dir}/*_encoder.pth')
            if len(existing_models)>1:
                existing_models.sort(key=os.path.getmtime, reverse=True)
                for model_epoch in existing_models[1:]:
                    os.remove(model_epoch)
                    os.remove(model_epoch.replace('_encoder.pth', '_decoder.pth'))
            best_model = 'model' + existing_models[0].split('model')[-1].split('_encoder')[0]

            # Load best model for evaluation
            model = autoencoder(args.encoder_name, args.lr, device, args.batch_size)
            model.load(f'{save_dir}/{best_model}')

            # Evaluate model
            print(f'Evaluating with {best_model} ...')
            
            ## distance and density results
            global_dist_dens_results = evaluate(test_data, model, batch_size=512, local=False)

            ## loss results
            loss_results = model.compute_loss(test_data)
            loss_results = {'loss': loss_results}

            # Save evaluation results
            key_values = {**loss_results, **global_dist_dens_results}
            keys = list(key_values.keys())
            values = np.array(list(key_values.values())).astype(np.float32)
            eval_results = pd.read_csv(results_dir) # read saved results again to avoid overwriting
            eval_results.loc[(eval_results['bslr']==bslr), keys] = values
            eval_results.loc[(eval_results['bslr']==bslr), 'model_used'] = best_model

            # Save evaluation results per dataset and model
            eval_results.to_csv(results_dir, index=False)
        print(f'--- {bslr} evaluation time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)) + ' ---')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
