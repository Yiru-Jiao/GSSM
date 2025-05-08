'''
This script reuses the UCD model to train a SVGP model with the SafeBaseline dataset.
'''

import os
import sys
import random
import time as systime
import torch
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_posterior_inference.inference_utils.utils_general import fix_seed, init_dl_program
from src_safety_evaluation.reuse_ucd.unified_conflit_detection import *
manual_seed = 131


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be 1,2 for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    return args


def main(args, manual_seed, model_path):
    initial_time = systime.time()
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    
    # Set the random seed
    if args.reproduction:
        args.seed = manual_seed # Fix the random seed for reproduction
    if args.seed is None:
        args.seed = random.randint(0, 1000)
    print(f'Random seed is set to {args.seed}')
    fix_seed(args.seed, deterministic=args.reproduction)
    
    # Initialize the deep learning program
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    # Read event data
    beta = 5
    initial_lr = 0.025
    batch_size = 2048
    num_qepochs = 500
    num_inducing_points = 100

    # Training
    existing_files = os.listdir(model_path)
    existing_files = [file for file in existing_files if file.endswith('.pth')]
    if len(existing_files) > 0:
        print('Model already trained. Exiting...')
    else:
        pipeline = train_val_test(device, num_inducing_points, 
                                  path_input='PreparedData/Segments/SafeBaseline/',
                                  path_output=model_path)
        pipeline.create_dataloader(batch_size, beta)
        print('Training...')
        pipeline.train_model(num_qepochs, initial_lr)

        print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    model_path = 'PreparedData/PosteriorInference/SafeBaseline/ucd/'
    os.makedirs(model_path, exist_ok=True)
    main(args, manual_seed, model_path)
