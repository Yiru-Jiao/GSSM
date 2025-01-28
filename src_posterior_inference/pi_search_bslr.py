'''
This script searches for suitable batch size and learning rate for the posterior inference model.
'''

import os
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
import time as systime
from inference_utils.utils_train_eval_test import set_experiments, train_val_test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    parser.add_argument('--reversed_list', type=int, default=0, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
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

    exp_config = set_experiments(stage=[1,2,4])
    if args.reversed_list:
        exp_config = exp_config[::-1]

    datasets = [exp[0] for exp in exp_config]
    encoder_combinations = [exp[1] for exp in exp_config]
    cross_attention_flag = [exp[2] for exp in exp_config]
    pretraining_flag = [exp[3] for exp in exp_config]

    os.makedirs(path_prepared + 'PosteriorInference/', exist_ok=True)
    if os.path.exists(path_prepared + 'PosteriorInference/bslr_search.csv'):
        bslr_search = pd.read_csv(path_prepared + 'PosteriorInference/bslr_search.csv')
    else:
        bslr_search = pd.DataFrame(columns=['dataset', 'encoder_selection', 'cross_attention', 'pretraining', 'initial_lr', 'batch_size', 'avg_val_loss'])
        bslr_search.to_csv(path_prepared + 'PosteriorInference/bslr_search.csv', index=False)

    for dataset, encoder_selection, cross_attention, pretrained_encoder in zip(datasets, encoder_combinations, cross_attention_flag, pretraining_flag):
        dataset_name = '_'.join(dataset)
        encoder_name = '_'.join(encoder_selection)
        cross_attention_name = '_'.join(cross_attention) if len(cross_attention)>0 else 'not_crossed'
        pretraining = 'pretrained' if pretrained_encoder else 'not_pretrained'

        initial_lr = 0.0001
        factor_range = range(5, 10) # 32, 64, 128, 256, 512
        if args.reversed_list:
            factor_range = factor_range[::-1]
        for factor in factor_range:
            sub_initial_time = systime.time()
            batch_size = 2**factor
            epochs = 6 * batch_size//32 # maintain the same number of gradient updates
            condition = (bslr_search['dataset']==dataset_name)&\
                        (bslr_search['encoder_selection']==encoder_name)&\
                        (bslr_search['cross_attention']==cross_attention_name)&\
                        (bslr_search['pretraining']==pretraining)&\
                        (bslr_search['initial_lr']==initial_lr)&\
                        (bslr_search['batch_size']==batch_size)
            if len(bslr_search[condition])>0 and not np.isnan(bslr_search.loc[condition, 'avg_val_loss'].values[0]):
                print(f"{encoder_name}, initial_lr: {initial_lr}, batch_size: {batch_size} already done.")
                continue
            print(f"{encoder_name}, initial_lr: {initial_lr}, batch_size: {batch_size} start training.")
            pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, cross_attention, pretrained_encoder)
            pipeline.create_dataloader(batch_size)
            pipeline.train_model(epochs, initial_lr, lr_schedule=False, verbose=2)
            avg_val_loss = pipeline.val_loss_log[-batch_size//32:].mean() # average over the same number of gradient updates
            bslr_search = pd.read_csv(path_prepared + 'PosteriorInference/bslr_search.csv')
            bslr_search.loc[len(bslr_search)] = [dataset_name, encoder_name, cross_attention_name, pretraining,
                                                    initial_lr, batch_size, avg_val_loss]
            bslr_search = bslr_search.sort_values(by=['dataset', 'encoder_selection', 'cross_attention', 'pretraining', 'batch_size', 'initial_lr'])
            bslr_search.to_csv(path_prepared + 'PosteriorInference/bslr_search.csv', index=False)
        print(f"{encoder_name}, initial_lr: {initial_lr}, batch_size: {batch_size} done, time elapsed: {systime.time()-sub_initial_time:.2f}s.")
    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    manual_seed = 131
    path_prepared = './PreparedData/'
    main(args, manual_seed, path_prepared)


