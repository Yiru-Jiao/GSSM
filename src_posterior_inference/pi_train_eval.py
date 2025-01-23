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
from inference_utils.utils_train_eval_test import train_val_test
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

    datasets = [['highD'],
                ['SafeBaseline'],
                ['INTERACTION'],
                ['Argoverse'],
                ['SafeBaseline']]
                # ['INTERACTION', 'highD'],
                # ['SafeBaseline', 'INTERACTION'],
                # ['SafeBaseline', 'highD'],
                # ['SafeBaseline', 'INTERACTION', 'highD']]
    encoder_combinations = [['current'],
                            ['current'],
                            ['current'],
                            ['current'],
                            ['current', 'environment']]
                            # ['current'],
                            # ['current'],
                            # ['current'],
                            # ['current']]
                            # ['current', 'environment', 'profiles']]
    cross_attention_flag = [[], [], [], [], []]
                            # ['first','last'], 
                            # ['first','middle','last']]
    if args.reversed_list:
        datasets = datasets[::-1]
        encoder_combinations = encoder_combinations[::-1]
        cross_attention_flag = cross_attention_flag[::-1]
    
    if os.path.exists(path_prepared + 'PosteriorInference/evaluation.csv'):
        evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
    else:
        evaluation = pd.DataFrame(columns=['dataset', 'encoder_selection', 'cross_attention', 'pretraining', 'initial_lr', 'batch_size', 'val_loss'])
    bslr_search = pd.read_csv(path_prepared + 'PosteriorInference/bslr_search.csv')
    for pretrained_encoder in [False, True]:
        pretraining = 'pretrained' if pretrained_encoder else 'not_pretrained'
        for dataset, encoder_selection, cross_attention in zip(datasets, encoder_combinations, cross_attention_flag):
            if (not 'highD' in dataset) and pretrained_encoder:
                continue 
            dataset_name = '_'.join(dataset)
            encoder_name = '_'.join(encoder_selection)
            cross_attention_name = '_'.join(cross_attention) if len(cross_attention)>0 else 'not_crossed'
            epochs = 300

            bslr = bslr_search[bslr_search['encoder_selection']==encoder_name].sort_values(by='avg_val_loss')
            batch_size = int(bslr['batch_size'].values[0])
            initial_lr = round(float(bslr['initial_lr'].values[0]), 4)

            condition = (evaluation['dataset']==dataset_name)&\
                        (evaluation['encoder_selection']==encoder_name)&\
                        (evaluation['cross_attention']==cross_attention_name)&\
                        (evaluation['pretraining']==pretraining)&\
                        (evaluation['initial_lr']==initial_lr)&\
                        (evaluation['batch_size']==batch_size)
            if len(evaluation[condition])>0 and not np.isnan(evaluation.loc[condition, 'val_loss'].values[0]):
                print(f"{dataset_name}, {encoder_name}, {cross_attention_name}, {pretraining}, initial_lr: {initial_lr}, batch_size: {batch_size} already done.")
                continue
            print(f"{dataset_name}, {encoder_name}, {cross_attention_name}, {pretraining}, initial_lr: {initial_lr}, batch_size: {batch_size} start training.")
            pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, cross_attention, pretrained_encoder)
            pipeline.create_dataloader(batch_size)
            if os.path.exists(pipeline.path_output + f'val_loss_log.csv'):
                print(f"Loading trained model: {dataset_name}, {encoder_name}, {cross_attention_name}, {pretraining}, initial_lr: {initial_lr}, batch_size: {batch_size}.")
                pipeline.load_model()
                val_loss = pd.read_csv(pipeline.path_output + 'val_loss_log.csv')
                val_loss = np.sort(val_loss['val_loss'].values[-5:])[1:4].mean()
            else:
                pipeline.train_model(epochs, initial_lr, verbose=5)
                val_loss = np.sort(pipeline.val_loss_log[-5:])[1:4].mean()
            evaluation.loc[len(evaluation)] = [dataset_name, encoder_name, cross_attention_name, pretraining,
                                               initial_lr, batch_size, val_loss]
            evaluation = evaluation.sort_values(by=['dataset', 'encoder_selection', 'cross_attention', 'pretraining'])
            evaluation.to_csv(path_prepared + 'PosteriorInference/evaluation.csv', index=False)
    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    manual_seed = 131
    path_prepared = './PreparedData/'
    main(args, manual_seed, path_prepared)


