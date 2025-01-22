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
path_processed = './ProcessedData/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_name', type=str, default='current', help='The encoder name to use for training and inference (defaults to current)')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    
    # Set default parameters
    if args.encoder_name == 'current':
        args.epochs = 1500
    elif args.encoder_name == 'environment':
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

    # Create the directory to save the evaluation results
    run_dir = f'{path_prepared}EncoderPretraining/{args.encoder_name}_autoencoder/trained_models/'
    results_dir = f'{path_prepared}EncoderPretraining/{args.encoder_name}_autoencoder/evaluation.csv'
    os.makedirs(run_dir, exist_ok=True)

    # Define metrics
    knn_metrics = ['mean_shared_neighbours', 'mean_dist_mrre', 'mean_trustworthiness', 'mean_continuity'] # kNN-based, averaged over various k
    
    if args.encoder_name == 'current':
        dataset_list = ['highD', 'SafeBaseline', 'INTERACTION', 'Argoverse', 'highD_SafeBaseline_INTERACTION_Argoverse']
        bslr_list = ['bs32_lr0.0001', 'bs64_lr0.0001', 'bs128_lr0.0001', 'bs256_lr0.0001']
    else:
        dataset_list = [None]
        bslr_list = ['bs64_lr0.001', 'bs128_lr0.001', 'bs256_lr0.001', 'bs512_lr0.001']
    
    if os.path.exists(results_dir):
        eval_results = pd.read_csv(results_dir)
    else:
        metrics = ['global_'+metric for metric in knn_metrics]
        if args.encoder_name == 'current':
            eval_results = pd.DataFrame(np.zeros((len(bslr_list)*len(dataset_list), 4), dtype=np.float32), columns=metrics)
            eval_results[['dataset', 'bslr']] = [[dataset, bslr] for dataset in dataset_list for bslr in bslr_list]
        else:
            eval_results = pd.DataFrame(np.zeros((len(bslr_list), 4), dtype=np.float32), columns=metrics)
            eval_results['bslr'] = bslr_list
        eval_results.to_csv(results_dir, index=False)
    
    # Iterate over different model sets
    verbose = 5 # update per n_epochs // (1+verbose*4) epoch
    for dataset in dataset_list:
        print('---- Loading data ----')
        if dataset is None:
            train_data, test_data = datautils.load_data([None], dataset_dir=path_prepared, feature=args.encoder_name)
        else:
            train_data, test_data = datautils.load_data(dataset.split('_'), dataset_dir=path_prepared, feature=args.encoder_name)
            test_data = test_data[np.random.choice(test_data.shape[0], 10000, replace=False)] # reduce test data size to avoid memory error

        for bslr in bslr_list:
            args.batch_size = int(bslr.split('_')[0].replace('bs',''))
            args.lr = float(bslr.split('_')[1].replace('lr',''))

            if args.encoder_name=='current':
                condition = (eval_results['dataset']==dataset)&(eval_results['bslr']==bslr)
                save_dir = os.path.join(run_dir, f'{dataset}/{bslr}')
            else:
                condition = (eval_results['bslr']==bslr)
                save_dir = os.path.join(run_dir, f'{bslr}')
            os.makedirs(save_dir, exist_ok=True)

            if eval_results[condition]['global_mean_shared_neighbours'].values[0]>0:
                final_epoch = eval_results[condition]['model_used'].values[0].split('epo')[0].split('_')[-1]
                print(f'--- {dataset} {bslr} has been trained until epoch {final_epoch}, skipping evaluation ---')
                continue

            start_time = systime.time()
            # Train model if not already trained
            if os.path.exists(f'{save_dir}/loss_log.csv'):
                print(f'--- {dataset} {bslr} has been trained, loading final model ---')
            else:
                # Create model
                model = autoencoder(args.encoder_name, args.lr, device, args.batch_size,
                                    after_epoch_callback = save_checkpoint_callback(save_dir, 0, unit='epoch'))
                scheduler = 'reduced'
                print(f'--- {dataset} {bslr} training with ReduceLROnPlateau scheduler ---')
                loss_log = model.fit(train_data, args.epochs, scheduler, verbose=verbose)
                # Save loss log
                loss_log = pd.DataFrame(loss_log, index=[f'epoch_{i}' for i in range(1, len(loss_log)+1)],
                                        columns=[f'iter_{i}' for i in range(1, len(loss_log[0])+1)])
                loss_log.to_csv(f'{save_dir}/loss_log.csv')
                print(f'Training time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - start_time)))
            
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
            global_dist_dens_results = evaluate(test_data, model, batch_size=256, local=False)

            ## loss results
            loss_results = model.compute_loss(test_data)
            loss_results = {'rmse_loss': loss_results}

            # Save evaluation results
            key_values = {**loss_results, **global_dist_dens_results}
            keys = list(key_values.keys())
            values = np.array(list(key_values.values())).astype(np.float32)
            eval_results = pd.read_csv(results_dir) # read saved results again to avoid overwriting
            if args.encoder_name=='current':
                condition = (eval_results['dataset']==dataset)&(eval_results['bslr']==bslr)
            else:
                condition = (eval_results['bslr']==bslr)
            eval_results.loc[condition, keys] = values
            eval_results.loc[condition, 'model_used'] = best_model

            # Save evaluation results per dataset and model
            eval_results.to_csv(results_dir, index=False)
            
    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    main(args)
