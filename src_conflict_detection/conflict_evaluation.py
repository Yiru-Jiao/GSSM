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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()


manual_seed = 131
path_processed = './ProcessedData/'
path_prepared = './PreparedData/'
path_result = './ResultData/'


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


def main(meta_both, events, args):
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
    meta_both = meta_both[(meta_both['event_category']!='SafeBaseline')&
                          (meta_both['ego_reconstructed'].astype(bool))&
                          (meta_both['surrounding_reconstructed'].astype(bool))]
    
    path_save = path_result + 'ConflictDetection/'
    os.makedirs(path_save, exist_ok=True)
    for event_cat in meta_both['event_category'].value_counts().index.values[::-1]:
        print(f'--- Evaluating {event_cat} ---')
        os.makedirs(path_save + f'{event_cat}/', exist_ok=True)
        if os.path.exists(path_save + f'{event_cat}/events.h5'):
            data = pd.read_hdf(path_save + f'{event_cat}/events.h5', key='data')
        else:
            data = read_data(event_cat)
            data.to_hdf(path_save + f'{event_cat}/events.h5', key='data', mode='w')
        assert data['event_id'].nunique() == len(meta_both[meta_both['event_category']==event_cat])

        model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
        encoder_combinations = model_evaluation['encoder_selection'].unique()
        for pretrained_encoder in [False, True]:
            pretraining = 'pretrained' if pretrained_encoder else 'not_pretrained'
            for encoder_name in encoder_combinations:
                encoder_selection = encoder_name.split('_')
                if os.path.exists(path_save + f'{event_cat}/{encoder_name}_{pretraining}.h5'):
                    print(f'{event_cat} has been evaluated by {encoder_name}-{pretraining} encoder.')
                    continue
                # Define and load trained model
                best_model = model_evaluation[(model_evaluation['encoder_selection']==encoder_name)&
                                              (model_evaluation['pretraining']==pretraining)].sort_values('test_loss')
                if len(best_model)==0:
                    print(f'No model is available for {encoder_name}-{pretraining} encoder.')
                    continue
                else:
                    best_model = best_model.iloc[0]
                batch_size = best_model['batch_size']
                initial_lr = best_model['initial_lr']
                model = define_model(device, path_prepared, encoder_selection, pretrained_encoder, batch_size, initial_lr)

                # Organise features for each event and target
                profiles_features = []
                current_features = []
                spacing_list = []
                event_id_list = []
                target_ids = data.index.get_level_values('target_id').unique()
                for target_id in tqdm(target_ids, desc='Target', ascii=True):
                    df = data.loc(axis=0)[target_id, :]
                    if len(df)<25: # skip if the target was detected for less than 2.5 seconds
                        continue
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

                mu, sigma, probability, max_intensity, _ = assess_conflict(states, model, device, output='all')
                record = pd.DataFrame(event_id_list, columns=['event_id','target_id','time'])
                record[['event_id','target_id']] = record[['event_id','target_id']].astype(int)
                record['proximity'] = spacing_list
                record['mu'] = mu
                record['sigma'] = sigma
                record['probability'] = probability
                record['intensity'] = max_intensity
                record.to_hdf(path_save + f'{event_cat}/{encoder_name}_{pretraining}.h5', key='data', mode='w')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)
    args = parse_args()

    # Load metadata and event information
    meta_both = pd.read_csv(path_processed + 'metadata_birdseye.csv')
    meta_both = meta_both.set_index('event_id')
    events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_table.csv').set_index('eventID')
    
    main(meta_both, events, args)
