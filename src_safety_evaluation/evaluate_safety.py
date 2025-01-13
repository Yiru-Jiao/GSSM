'''
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
from validation_utils.utils_detection import *
import validation_utils.TwoDimTTC as TwoDimTTC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
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

    # Define scaler and one-hot encoder for normalisation
    current_scaler = get_scaler(path_prepared, feature='current')
    profiles_scaler = get_scaler(path_prepared, feature='profiles')
    environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
    one_hot_encoder = create_categorical_encoder(events, environment_feature_names)

    # Evaluate for each event category
    path_save = path_result + 'ConflictDetection/'
    event_categories = os.listdir(path_save)
    for event_cat in event_categories:
        print(f'--- Evaluating {event_cat} ---')
        event_meta = pd.read_csv(path_save + f'{event_cat}/event_meta.csv').set_index('event_id')
        data = pd.read_hdf(path_save + f'{event_cat}/event_data.h5', key='data')
        assert np.all(np.isin(data['event_id'].unique(), event_meta.index.values))

        # Self-supervised traffic safety evaluation
        model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
        for model_id in range(len(model_evaluation)):
            encoder_name = model_evaluation.iloc[model_id]['encoder_selection']
            encoder_selection = encoder_name.split('_')
            pretraining = model_evaluation.iloc[model_id]['pretraining']
            pretrained_encoder = True if pretraining=='pretrained' else False
            cross_attention_name = model_evaluation.iloc[model_id]['cross_attention']
            cross_attention = cross_attention_name.split('_') if cross_attention_name!='not_crossed' else []

            if os.path.exists(path_save + f'{event_cat}/{pretraining}/{encoder_name}_{cross_attention_name}.h5'):
                print(f'{event_cat} has been evaluated by {encoder_name}-{cross_attention_name}-{pretraining} encoder.')
                continue
            # Define and load trained model
            model = define_model(device, path_prepared, encoder_selection, cross_attention, pretrained_encoder)

            # Organise features for each event and target
            profiles_features = []
            current_features = []
            spacing_list = []
            event_id_list = []
            target_ids = event_meta[event_meta['duration_enough']].index.values
            for target_id in tqdm(target_ids, desc='Target', ascii=True):
                df = data.loc(axis=0)[target_id, :]
                segmented_features = get_context_representations(df, current_scaler, profiles_scaler)
                profiles_features.append(segmented_features[0])
                current_features.append(segmented_features[1])
                spacing_list.append(segmented_features[2])
                event_id_list.append(segmented_features[3])
            profiles_features = np.concatenate(profiles_features, axis=0)
            current_features = np.concatenate(current_features, axis=0)
            spacing_list = np.concatenate(spacing_list, axis=0)
            event_id_list = np.concatenate(event_id_list, axis=0)
            assert profiles_features.shape == (len(spacing_list), 20, 3)
            
            states = []
            if 'current' in encoder_selection:
                states.append(current_features)
            if 'environment' in encoder_selection:
                environment_features = events.loc[event_id_list[:,0], environment_feature_names].fillna('Unknown')
                environment_features = one_hot_encoder.transform(environment_features.values)
                states.append(environment_features)
            if 'profiles' in encoder_selection:
                states.append(profiles_features)
            if len(states) == 1: # only current features
                states = [states[0], spacing_list]
            else:
                states = [tuple(states), spacing_list]

            mu, sigma, max_intensity = SSSE(states, model, device)
            results = pd.DataFrame(event_id_list, columns=['event_id','target_id','time'])
            results[['event_id','target_id']] = results[['event_id','target_id']].astype(int)
            results['proximity'] = spacing_list
            results['mu'] = mu
            results['sigma'] = sigma
            results['intensity'] = max_intensity
            results.to_hdf(path_save + f'{event_cat}/{pretraining}/{encoder_name}_{cross_attention_name}.h5', key='data', mode='w')

        # Two-dimensional time-to-collision (2D-TTC) and Deceleration Rate to Avoid Collision (DRAC)
        rename_columns = dict()
        for column in data.columns:
            if '_ego' in column:
                rename_columns[column] = column.replace('_ego','_i')
            elif '_sur' in column:
                rename_columns[column] = column.replace('_sur','_j')
        results = data.rename(columns=rename_columns).copy()
        results['vx_i'] = results['v_i']*results['hx_i']
        results['vy_i'] = results['v_i']*results['hy_i']
        results['vx_j'] = results['v_j']*results['hx_j']
        results['vy_j'] = results['v_j']*results['hy_j']

        veh_dimensions = event_meta[['ego_width','ego_length','target_width','target_length']].copy()
        condition = event_meta[['target_width','target_length']].isna().any(axis=1)
        veh_dimensions.loc[condition, ['target_width','target_length']] = event_meta.loc[condition, ['other_width','other_length']].values
        results[['width_i','length_i','width_j','length_j']] = veh_dimensions.loc[results['event_id'].values].values

        results['TTC'] = TwoDimTTC.TTC(results, 'values')

        results['s_box'] = TwoDimTTC.CurrentD(results, 'values')
        results.loc[results['s_box']<1e-6, 's_box'] = 1e-6
        results['delta_v'] = np.sqrt((results['vx_i']-results['vx_j'])**2 + (results['vy_i']-results['vy_j'])**2)
        results['DRAC'] = results['delta_v']**2 / 2 / results['s_box']
        results.loc[results['v_i']<=results['v_j'], 'DRAC'] = 0.

        results = results[['event_id','target_id','time','width_i','length_i','width_j','length_j','s_box', 'delta_v', 'TTC', 'DRAC']]
        results.to_hdf(path_save + f'{event_cat}/TTC_DRAC.h5', key='data', mode='w')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    manual_seed = 131
    path_prepared = './PreparedData/'
    path_result = './ResultData/'

    # Load event information to create one-hot encoder later
    events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_table.csv').set_index('eventID')
    
    main(args, events, manual_seed, path_prepared, path_result)
