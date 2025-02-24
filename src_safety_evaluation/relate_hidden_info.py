'''
This script 
'''

import os
import sys
import random
import time as systime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.preprocessing import OneHotEncoder
from validation_utils.utils_features import *
from src_safety_evaluation.validation_utils.utils_evaluation import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_encoder_pretraining.ssrl_utils.utils_eval import *
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    return args


def create_categorical_encoder(events, environment_feature_names):
    categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    events.loc[events['surfaceCondition']=='Other','surfaceCondition'] = 'Unknown'
    data2fit = events[environment_feature_names].fillna('Unknown')
    data2fit = data2fit.loc[(data2fit!='Unknown').all(axis=1)]
    categorical_encoder.fit(data2fit.values)
    return categorical_encoder


def main(args, events, manual_seed, path_prepared, path_result):
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

    event_categories = sorted(os.listdir(path_result + 'EventData/'))
    path_save = path_result + 'EventData/'
    model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
    os.makedirs(path_save, exist_ok=True)

    profiles_features_list = []
    current_features_list = []
    spacing_list_list = []
    event_id_list_list = []
    # For each event category
    for event_cat in event_categories:
        print(f'--- Evaluating event category: {event_cat} ---')
        event_featurs = np.load(path_save + f'{event_cat}/event_features.npz')
        profiles_features = event_featurs['profiles']
        current_features = event_featurs['current']
        spacing_list = event_featurs['spacing']
        event_id_list = event_featurs['event_id']

        # Save attention weights
        model2use = ''
        for model_id in range(len(model_evaluation)):
            dataset_name = model_evaluation.iloc[model_id]['dataset']
            dataset = dataset_name.split('_')
            encoder_name = model_evaluation.iloc[model_id]['encoder_selection']
            encoder_selection = encoder_name.split('_')
            cross_attention_name = model_evaluation.iloc[model_id]['cross_attention']
            cross_attention = cross_attention_name.split('_') if cross_attention_name!='not_crossed' else []
            pretraining = model_evaluation.iloc[model_id]['pretraining']
            pretrained_encoder = True if pretraining=='pretrained' else False
            model_name = f'{dataset_name}_{encoder_name}_{cross_attention_name}_{pretraining}'
            if model_name==model2use:
                break
        print(f'--- Evaluating {model_name} ---')

        if os.path.exists(path_save + f'{event_cat}/attention_weights.npz'):
            print(f'--- Attention weights have been saved ---')
            continue

        # Define scaler and one-hot encoder for normalisation
        # current_scaler = get_scaler(dataset, path_prepared, feature=encoder_selection[0])
        # profiles_scaler = get_scaler(dataset, path_prepared, feature='profiles')
        if 'environment' in encoder_selection:
            environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
            one_hot_encoder = create_categorical_encoder(events, environment_feature_names)

        # Define and load trained model
        model = define_model(device, path_prepared, dataset, encoder_selection, cross_attention, pretrained_encoder)

        states = []
        if 'current' in encoder_selection:
            #? states.append(current_scaler.transform(current_features[:,:-1]))
            states.append(np.concatenate([current_features[:,:-2], current_features[:,-1:]], axis=1))
        if 'current+acc' in encoder_selection:
            #? states.append(current_scaler.transform(current_features))
            states.append(current_features)
        if 'environment' in encoder_selection:
            environment_features = events.loc[event_id_list[:,0], environment_feature_names].fillna('Unknown')
            environment_features = one_hot_encoder.transform(environment_features.values)
            states.append(environment_features)
        if 'profiles' in encoder_selection:
            # states.append(profiles_scaler.transform(profiles_features.reshape(-1, 4)).reshape(profiles_features.shape))
            states.append(profiles_features)
        if len(states) == 1: # only current features
            states = [states[0], spacing_list]
        else:
            states = [tuple(states), spacing_list]

        mu, sigma, max_intensity = SSSE(states, model, device)
        results = pd.DataFrame(event_id_list, columns=['event_id','target_id','time'])



    # Evaluate structure preservation
    profiles_features = np.concatenate(profiles_features_list, axis=0)
    current_features = np.concatenate(current_features_list, axis=0)
    spacing_list = np.concatenate(spacing_list_list, axis=0)
    event_id_list = np.concatenate(event_id_list_list, axis=0)

    for model_id in range(len(model_evaluation)):
        if model_evaluation.iloc[model_id]['global_mean_shared_neighbours'] > 0:
            print(f'--- {model_name} has been evaluated ---')
            continue

        dataset_name = model_evaluation.iloc[model_id]['dataset']
        dataset = dataset_name.split('_')
        encoder_name = model_evaluation.iloc[model_id]['encoder_selection']
        encoder_selection = encoder_name.split('_')
        cross_attention_name = model_evaluation.iloc[model_id]['cross_attention']
        cross_attention = cross_attention_name.split('_') if cross_attention_name!='not_crossed' else []
        pretraining = model_evaluation.iloc[model_id]['pretraining']
        pretrained_encoder = True if pretraining=='pretrained' else False
        model_name = f'{dataset_name}_{encoder_name}_{cross_attention_name}_{pretraining}'
        print(f'--- Evaluating {model_name} ---')

        # Define scaler and one-hot encoder for normalisation
        # current_scaler = get_scaler(dataset, path_prepared, feature=encoder_selection[0])
        # profiles_scaler = get_scaler(dataset, path_prepared, feature='profiles')
        if 'environment' in encoder_selection:
            environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
            one_hot_encoder = create_categorical_encoder(events, environment_feature_names)

        # Define and load trained model
        model = define_model(device, path_prepared, dataset, encoder_selection, cross_attention, pretrained_encoder)

        states = []
        if 'current' in encoder_selection:
            #? states.append(current_scaler.transform(current_features[:,:-1]))
            states.append(np.concatenate([current_features[:,:-2], current_features[:,-1:]], axis=1))
        if 'current+acc' in encoder_selection:
            #? states.append(current_scaler.transform(current_features))
            states.append(current_features)
        if 'environment' in encoder_selection:
            environment_features = events.loc[event_id_list[:,0], environment_feature_names].fillna('Unknown')
            environment_features = one_hot_encoder.transform(environment_features.values)
            states.append(environment_features)
        if 'profiles' in encoder_selection:
            # states.append(profiles_scaler.transform(profiles_features.reshape(-1, 4)).reshape(profiles_features.shape))
            states.append(profiles_features)
        if len(states) == 1: # only current features
            states = [states[0], spacing_list]
        else:
            states = [tuple(states), spacing_list]

        # Evaluate model
        if isinstance(states[0]) is tuple:
            testdata = states[0][0]
        else:
            testdata = states[0]
        global_dist_dens_results = evaluate(testdata, model, batch_size=512, local=False, states=states)

        # Save evaluation results
        keys = list(global_dist_dens_results.keys())
        values = np.array(list(global_dist_dens_results.values())).astype(np.float32)
        model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv') # read saved results again to avoid overwriting
        model_evaluation.loc[(model_evaluation['dataset']==dataset_name)&
                                (model_evaluation['encoder_selection']==encoder_name)&
                                (model_evaluation['cross_attention']==cross_attention_name)&
                                (model_evaluation['pretraining']==pretraining), keys] = values
        model_evaluation.to_csv(path_prepared + 'PosteriorInference/evaluation.csv', index=False)

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    manual_seed = 131
    path_prepared = './PreparedData/'
    path_result = './ResultData/'

    # Load event information to create one-hot encoder later
    events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_Table.csv').set_index('eventID')
    
    main(args, events, manual_seed, path_prepared, path_result)
