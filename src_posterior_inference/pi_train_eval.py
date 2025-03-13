'''
This script trains and evaluates the posterior inference model in various configurations.
'''

import os
import sys
import random
import time as systime
import numpy as np
import pandas as pd
import torch
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_posterior_inference.inference_utils.utils_train_eval_test import set_experiments, train_val_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--stage', type=int, default=None, help='The experiment stage to run (defaults to None for all stages)')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    parser.add_argument('--reversed_list', type=int, default=0, help='Whether to reverse the model list (defaults to False), useful for running parallel jobs')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    args.reversed_list = bool(args.reversed_list)
    return args


def main(args, manual_seed, path_prepared):
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

    if args.stage is None:
        exp_config = set_experiments(stage=[1,2,3,4])
    else:
        exp_config = set_experiments(stage=[args.stage])
    if args.reversed_list:
        exp_config = exp_config[::-1]

    datasets = [exp[0] for exp in exp_config]
    encoder_combinations = [exp[1] for exp in exp_config]
    pretraining_flag = [exp[2] for exp in exp_config]

    os.makedirs(path_prepared + 'PosteriorInference/', exist_ok=True)
    if os.path.exists(path_prepared + 'PosteriorInference/evaluation.csv'):
        evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
    else:
        evaluation = pd.DataFrame(columns=['dataset', 'encoder_selection', 'pretraining', 'val_loss', 'model_size'])
        evaluation.to_csv(path_prepared + 'PosteriorInference/evaluation.csv', index=False)
    for dataset, encoder_selection, pretrained_encoder in zip(datasets, encoder_combinations, pretraining_flag):
        dataset_name = '_'.join(dataset)
        encoder_name = '_'.join(encoder_selection)
        if pretrained_encoder==False:
            pretraining = 'not_pretrained'
        elif pretrained_encoder==True:
            pretraining = 'pretrained'
        elif pretrained_encoder=='all':
            pretraining = 'pretrained_all'

        initial_lr = 0.0002
        batch_size = 512
        epochs = 300
        
        condition = (evaluation['dataset']==dataset_name)&\
                    (evaluation['encoder_selection']==encoder_name)&\
                    (evaluation['pretraining']==pretraining)
        if len(evaluation[condition])>0 and not np.isnan(evaluation.loc[condition, 'val_loss'].values[0]):
            print(f"Already done {dataset_name}, {encoder_name}, {pretraining}")
            continue
        else:
            print(f"Start training {dataset_name}, {encoder_name}, {pretraining}")
        pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, pretrained_encoder)
        pipeline.create_dataloader(batch_size)
        if os.path.exists(pipeline.path_output + f'loss_log.csv'):
            print(f"Loading trained model: {dataset_name}, {encoder_name}, {pretraining}")
            pipeline.load_model()
            val_loss = pd.read_csv(pipeline.path_output + 'loss_log.csv')
            val_loss = np.sort(val_loss['val_loss'].values[-5:])[1:4].mean()
        else:
            pipeline.train_model(epochs, initial_lr, lr_schedule=True, verbose=5)
            val_loss = np.sort(pipeline.val_loss_log[-5:])[1:4].mean()
        model_size = sum(p.numel() for p in pipeline.model.parameters())
        evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv') # Reload the evaluation file to make sure updated
        columns = ['dataset', 'encoder_selection', 'pretraining', 'val_loss', 'model_size']
        evaluation.loc[len(evaluation), columns] = [dataset_name, encoder_name, pretraining, val_loss, model_size]
        evaluation = evaluation.sort_values(by=['dataset', 'encoder_selection', 'pretraining'])
        evaluation.to_csv(path_prepared + 'PosteriorInference/evaluation.csv', index=False)
    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    manual_seed = 131
    path_prepared = 'PreparedData/'
    main(args, manual_seed, path_prepared)


