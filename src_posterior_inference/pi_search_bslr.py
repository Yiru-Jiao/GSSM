'''

'''

import os
import sys
import random
import time as systime
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
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
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

    encoder_combinations = [['current'],
                            ['current', 'environment'],
                            ['current', 'environment', 'profiles'],
                            ['current', 'environment', 'profiles'],
                            ['current', 'environment', 'profiles'],
                            ['current', 'environment', 'profiles'],
                            ['current', 'environment', 'profiles']]
    cross_attention_flag = [[], [], [], 
                            ['first'], 
                            ['last'], 
                            ['first','last'], 
                            ['first','middle','last']]

    if os.path.exists(path_prepared + 'PosteriorInference/bslr_search.csv'):
        bslr_search = pd.read_csv(path_prepared + 'PosteriorInference/bslr_search.csv')
    else:
        bslr_search = pd.DataFrame(columns=['encoder_selection', 'cross_attention', 'pretraining', 'initial_lr', 'batch_size', 'val_loss'])
    for pretrained_encoder in [False, True]:
        pretraining = 'pretrained' if pretrained_encoder else 'not_pretrained'
        for encoder_selection, cross_attention in zip(encoder_combinations, cross_attention_flag):
            encoder_flag = '_'.join(encoder_selection)
            cross_flag = '_'.join(cross_attention) if len(cross_attention)>0 else 'not_crossed'
            initial_lr = 0.001
            if 'profiles' in encoder_selection:
                epochs = 10
            else:
                epochs = 30
            for factor in range(5, 11):
                batch_size = 2**factor # 32, 64, 128, 256, 512, 1024
                condition = (bslr_search.encoder_selection==encoder_flag)&\
                            (bslr_search.cross_attention==cross_flag)&\
                            (bslr_search.pretraining==pretraining)&\
                            (bslr_search.initial_lr==initial_lr)&\
                            (bslr_search.batch_size==batch_size)
                if len(bslr_search[condition])>0 and bslr_search.loc[condition, 'val_loss'].values[0]<1:
                    print(f"{encoder_flag}, {cross_flag}, {pretraining}, initial_lr: {initial_lr}, batch_size: {batch_size} already done.")
                    continue
                print(f"{encoder_flag}, {cross_flag}, {pretraining}, initial_lr: {initial_lr}, batch_size: {batch_size} start training.")
                pipeline = train_val_test(device, path_prepared, encoder_selection, cross_attention, pretrained_encoder)
                pipeline.create_dataloader(batch_size)
                pipeline.train_model(epochs, initial_lr, lr_schedule=False, verbose=5)
                val_loss = pipeline.val_loop()
                bslr_search.loc[len(bslr_search)] = [encoder_flag, cross_flag, pretraining,
                                                     initial_lr, batch_size, val_loss]
                bslr_search = bslr_search.sort_values(by=['encoder_selection', 'cross_attention', 'pretraining', 'initial_lr', 'batch_size'])
                bslr_search.to_csv(path_prepared + 'PosteriorInference/bslr_search.csv', index=False)
    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    manual_seed = 131
    path_prepared = './PreparedData/'
    main(args, manual_seed, path_prepared)


