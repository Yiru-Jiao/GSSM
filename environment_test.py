'''
This script is used to test the environment setup. 
It checks if all the required libraries are installed and if a GPU is available. 
It also checks if the random seeds are fixed properly.
'''

import os
import sys
import glob
import torch
import random
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import time as systime
from datetime import datetime
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

# src_trajectory_reconstruction
from matplotlib.backends.backend_pdf import PdfPages
from src_trajectory_reconstruction.reconstruction_utils.utils_ego_sur import *
from src_trajectory_reconstruction.reconstruction_utils.utils_ekf import reconstruct_ego, reconstruct_surrounding
print('--- All the imports in src_trajectory_reconstruction are successful ---')

# src_data_preparation
import multiprocessing
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from src_data_preparation.represent_utils.coortrans import coortrans
print('--- All the imports in src_data_preparation are successful ---')

# src_encoder_pretraining
from tslearn.metrics import dtw, dtw_path
from src_encoder_pretraining.clt_model import spclt
from src_encoder_pretraining.modules.loss_utils import *
from src_encoder_pretraining.ae_model import autoencoder
from src_encoder_pretraining.modules.regularizers import *
from src_encoder_pretraining.ssrl_utils.utils_eval import *
from src_encoder_pretraining.modules import encoder, losses
from src_encoder_pretraining.ssrl_utils.utils_general import *
import src_encoder_pretraining.ssrl_utils.utils_data as datautils
from src_encoder_pretraining.ssrl_utils.utils_paramsearch import *
from src_encoder_pretraining.ssrl_utils.utils_distance_matrix import *
from src_encoder_pretraining.modules.measures import MeasureCalculator
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from src_encoder_pretraining.ssrl_utils.utils_distance_matrix import get_EUC
from src_posterior_inference.inference_utils.modules import current_encoder, environment_encoder
print('--- All the imports in src_encoder_pretraining are successful ---')

# src_posterior_inference
print('--- All the imports in src_posterior_inference are successful ---')

# src_conflict_detection
from src_safety_evaluation.validation_utils.utils_evaluation import *
from src_safety_evaluation.validation_utils.utils_features import *
print('--- All the imports in src_safety_evaluation are successful ---')


def main():
    manual_seed = 131

    print('--- All the imports are successful ---')
    
    print(f'--- Available cores: {torch.get_num_threads()} available gpus: {torch.cuda.device_count()} ---')
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    print(f'--- Pytorch version: {torch.__version__}, Available threads: {os.cpu_count()} ---')
    
    fix_seed(manual_seed, deterministic=True)  # Below random values in comments are results in the author's machine
    print(f'Random seed fixed to be {manual_seed}, testing...')
    print('Python random test:', random.random()) # 0.3154351888034451
    print('Numpy random test:', np.random.rand()) # 0.7809038987924661
    print('Torch random test:', torch.rand(1).item()) # 0.39420515298843384
    if torch.cuda.is_available():
        print('Cudnn random test:', torch.rand(1, device='cuda').item()) # 0.5410190224647522

    print('--- Run again to see if the random values are the same ---')

    sys.exit(0)

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
