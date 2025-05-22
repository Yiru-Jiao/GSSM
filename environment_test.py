'''
This script is used to test the environment setup. 
It checks if all the required libraries are installed and if a GPU is available. 
It also checks if the random seeds are fixed properly.
'''

# Standard libraries
import os
import gc
import sys
import glob
import torch
import random
import argparse
import warnings
import time as systime
from datetime import datetime

# Third-party libraries
import gpytorch
import numpy as np
import pandas as pd
from torch import nn
import scipy.special
from tqdm import tqdm
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

# src_trajectory_reconstruction
# from matplotlib.backends.backend_pdf import PdfPages
from src_trajectory_reconstruction.reconstruction_utils.utils_ekf import *
# from src_trajectory_reconstruction.reconstruction_utils.utils_ego_sur import *
print('--- All the imports in src_trajectory_reconstruction are successful ---')

# src_data_preparation
import multiprocessing
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from src_data_preparation.represent_utils.coortrans import coortrans
from src_data_preparation.represent_utils.get_heading_highD import *
from src_data_preparation.represent_utils.utils_data_segmentation import *
print('--- All the imports in src_data_preparation are successful ---')

# src_posterior_inference
from src_posterior_inference.model import *
from src_posterior_inference.inference_utils.modules import *
from src_posterior_inference.inference_utils.utils_data import *
from src_posterior_inference.inference_utils.utils_general import *
from src_posterior_inference.inference_utils.utils_train_eval_test import *
print('--- All the imports in src_posterior_inference are successful ---')

# src_conflict_detection
from src_safety_evaluation.validation_utils.SSMsOnPlane import *
from src_safety_evaluation.validation_utils.utils_features import *
from src_safety_evaluation.validation_utils.EmergencyIndex import *
from src_safety_evaluation.validation_utils.utils_evaluation import *
from src_safety_evaluation.validation_utils.utils_attribution import *
from src_safety_evaluation.validation_utils.utils_eval_metrics import *
print('--- All the imports in src_safety_evaluation are successful ---')

# visualization
# from src_visualisation.visual_utils.utils_tabfig import *
# from src_visualisation.visual_utils.utils_dynamic import *
# print('--- All the imports in src_visualisation are successful ---')


def main():
    manual_seed = 131
    
    print(f'--- Available cores: {torch.get_num_threads()} available gpus: {torch.cuda.device_count()} ---')
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    print(f'--- Pytorch version: {torch.__version__}, Available threads: {os.cpu_count()} ---')
    
    fix_seed(manual_seed, deterministic=True)  # Below random values in comments are results in the author's machine
    print(f'Random seed fixed to be {manual_seed}, testing...')
    print('Python random test:', random.random()) # 0.3154351888034451
    print('Numpy random test:', np.random.rand()) # 0.650153605471749
    print('Torch random test:', torch.rand(1).item()) # 0.43932265043258667
    if torch.cuda.is_available():
        print('Cudnn random test:', torch.rand(1, device='cuda').item()) # 0.6956863403320312

    print('--- Run again to see if the random values are the same ---')

    sys.exit(0)

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
