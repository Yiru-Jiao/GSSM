'''
This script applies different models to evaluate the safety of events.
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()
from src_safety_evaluation.validation_utils.utils_evaluation import *
from src_safety_evaluation.validation_utils.EmergencyIndex import get_EI
from src_safety_evaluation.validation_utils.SSMsOnPlane import longitudinal_ssms, two_dimensional_ssms


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


def set_veh_dimensions(event_meta, avg_width, avg_length):
    veh_dimensions = event_meta[['ego_width','ego_length','target_width','target_length']].copy()
    condition = event_meta[['target_width','target_length']].isna().any(axis=1)
    veh_dimensions.loc[condition, ['target_width','target_length']] = event_meta.loc[condition, ['other_width','other_length']].values
    for var in ['ego_width','ego_length','target_width','target_length']:
        veh_dimensions.loc[veh_dimensions[var].isna(), var] = avg_width if 'width' in var else avg_length
    return veh_dimensions


def evaluate(eval_func, model, eval_config, eval_efficiency, results, path_save):
    time_start = systime.time()
    results = eval_func(results, **eval_config)
    time_end = systime.time()
    eval_efficiency.loc[len(eval_efficiency)] = [model, time_end-time_start, results['target_id'].nunique(), len(results)]
    eval_efficiency.to_csv(path_save + 'EvaluationEfficiency.csv', index=False)
    return results, eval_efficiency


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

    # Load/save event features
    event_categories = sorted(os.listdir(path_result + 'EventData/'))
    event_meta = pd.concat([pd.read_csv(path_result + f'EventData/{event_cat}/event_meta.csv') for event_cat in event_categories], ignore_index=True).set_index('event_id')
    avg_width = np.nanmean(event_meta['ego_width'].values)
    avg_length = np.nanmean(event_meta['ego_length'].values)
    for event_cat in event_categories:
        if os.path.exists(path_result + f'EventData/{event_cat}/event_features.npz'):
            print(f'Loading event features for {event_cat}.')
        else:
            print(f'Saving event features for {event_cat}.')
            data = pd.read_hdf(path_result + f'EventData/{event_cat}/event_data.h5', key='data')
            event_meta = pd.read_csv(path_result + f'EventData/{event_cat}/event_meta.csv').set_index('event_id')
            assert np.all(np.isin(data['event_id'].unique(), event_meta.index.values))
            veh_dimensions = set_veh_dimensions(event_meta, avg_width, avg_length)
            # Organise features for each event and target
            profiles_features = []
            current_features = []
            spacing_list = []
            event_id_list = []
            target_ids = data[data['event_id'].isin(event_meta[event_meta['duration_enough']].index.values)].index.unique(level='target_id').values
            for target_id in tqdm(target_ids, desc='Target', position=0, dynamic_ncols=False, ascii=True, miniters=min(len(target_ids)//10, 150)):
                df = data.loc(axis=0)[target_id, :]
                if len(df)<25: # skip if the target was detected for less than 2.5 seconds
                    continue
                segmented_features = get_context_representations(df, veh_dimensions.loc[df['event_id'].values[0]])
                profiles_features.append(segmented_features[0]) # will need normalisation when being used
                current_features.append(segmented_features[1]) # will need normalisation when being used
                spacing_list.append(segmented_features[2])
                event_id_list.append(segmented_features[3])
            profiles_features = np.concatenate(profiles_features, axis=0)
            current_features = np.concatenate(current_features, axis=0)
            spacing_list = np.concatenate(spacing_list, axis=0)
            event_id_list = np.concatenate(event_id_list, axis=0)
            assert profiles_features.shape[0] == len(spacing_list) and profiles_features.shape[1] == 20
            # Save the features
            np.savez(path_result + f'EventData/{event_cat}/event_features.npz', 
                     profiles=profiles_features, 
                     current=current_features, 
                     spacing=spacing_list, 
                     event_id=event_id_list)

    # Read event features
    path_save = path_result + 'EventEvaluation/'
    os.makedirs(path_save, exist_ok=True)

    data = pd.concat([pd.read_hdf(path_result + f'EventData/{event_cat}/event_data.h5', key='data') for event_cat in event_categories]).reset_index()
    event_meta = pd.concat([pd.read_csv(path_result + f'EventData/{event_cat}/event_meta.csv') for event_cat in event_categories], ignore_index=True).set_index('event_id')
    veh_dimensions = set_veh_dimensions(event_meta, avg_width, avg_length)
    
    profiles_features = []
    current_features = []
    spacing_list = []
    event_id_list = []
    for event_cat in event_categories:
        event_featurs = np.load(path_result + f'EventData/{event_cat}/event_features.npz')
        profiles_features.append(event_featurs['profiles'])
        current_features.append(event_featurs['current'])
        spacing_list.append(event_featurs['spacing'])
        event_id_list.append(event_featurs['event_id'])
    profiles_features = np.concatenate(profiles_features, axis=0)
    current_features = np.concatenate(current_features, axis=0)
    spacing_list = np.concatenate(spacing_list, axis=0)
    event_id_list = np.concatenate(event_id_list, axis=0)

    # Safety evaluation
    if os.path.exists(path_save + 'EvaluationEfficiency.csv'):
        eval_efficiency = pd.read_csv(path_save + 'EvaluationEfficiency.csv', dtype={'model_name':str,'time':float,'num_targets':int,'num_moments':int})
    else:
        eval_efficiency = pd.DataFrame(columns=['model_name','time','num_targets','num_moments'])
        eval_efficiency.to_csv(path_save + 'EvaluationEfficiency.csv', index=False)

    # 1D SSMs adapted to 2D
    if os.path.exists(path_save + f'TTC_DRAC_MTTC_PSD.h5'):
        print(f'The events has been evaluated by TTC, DRAC, MTTC, and PSD.')
    else:
        print('--- Evaluating with TTC, DRAC, MTTC, and PSD ---')
        results = data.merge(pd.DataFrame(event_id_list, columns=['event_id','target_id','time']),
                             on=['event_id','target_id','time'], how='inner')
        rename_columns = dict()
        for column in results.columns:
            if '_ego' in column:
                rename_columns[column] = column.replace('_ego','_i')
            elif '_sur' in column:
                rename_columns[column] = column.replace('_sur','_j')
        results = results.rename(columns=rename_columns)
        results['vx_i'] = results['v_i']*results['hx_i']
        results['vy_i'] = results['v_i']*results['hy_i']
        results['vx_j'] = results['v_j']*results['hx_j']
        results['vy_j'] = results['v_j']*results['hy_j']
        results[['width_i','length_i','width_j','length_j']] = veh_dimensions.loc[results['event_id'].values].values

        results, eval_efficiency = evaluate(longitudinal_ssms.TTC, 'TTC',
                                            {'toreturn':'dataframe'},
                                            eval_efficiency, results, path_save)
        
        results, eval_efficiency = evaluate(longitudinal_ssms.DRAC, 'DRAC',
                                            {'toreturn':'dataframe'},
                                            eval_efficiency, results, path_save)
        
        results, eval_efficiency = evaluate(longitudinal_ssms.MTTC, 'MTTC',
                                            {'toreturn':'dataframe'},
                                            eval_efficiency, results, path_save)
        
        results, eval_efficiency = evaluate(longitudinal_ssms.PSD, 'PSD',
                                            {'toreturn':'dataframe', 'braking_dec': 5.5},
                                            eval_efficiency, results, path_save)

        results['s_box'] = longitudinal_ssms.CurrentD(results, 'values')
        results = results[['event_id','target_id','time','width_i','length_i','width_j','length_j','acc_i','s_box', 'TTC', 'DRAC', 'MTTC', 'PSD']]
        results.to_hdf(path_save + f'TTC_DRAC_MTTC_PSD.h5', key='data', mode='w')

    # 2D SSMs
    if os.path.exists(path_save + f'TAdv_TTC2D_ACT_EI.h5'):
        print(f'The events has been evaluated by TAdv, TTC2D, ACT, and EI.')
    else:
        print('--- Evaluating with TAdv, TTC2D, ACT, and EI ---')
        results = data.merge(pd.DataFrame(event_id_list, columns=['event_id','target_id','time']),
                             on=['event_id','target_id','time'], how='inner')
        rename_columns = dict()
        for column in results.columns:
            if '_ego' in column:
                rename_columns[column] = column.replace('_ego','_i')
            elif '_sur' in column:
                rename_columns[column] = column.replace('_sur','_j')
        results = results.rename(columns=rename_columns)
        results['vx_i'] = results['v_i']*results['hx_i']
        results['vy_i'] = results['v_i']*results['hy_i']
        results['vx_j'] = results['v_j']*results['hx_j']
        results['vy_j'] = results['v_j']*results['hy_j']
        results[['width_i','length_i','width_j','length_j']] = veh_dimensions.loc[results['event_id'].values].values
        
        results, eval_efficiency = evaluate(two_dimensional_ssms.TAdv, 'TAdv',
                                            {'toreturn':'dataframe'},
                                            eval_efficiency, results, path_save)
        
        results, eval_efficiency = evaluate(two_dimensional_ssms.TTC2D, 'TTC2D',
                                            {'toreturn':'dataframe'},
                                            eval_efficiency, results, path_save)
        
        results, eval_efficiency = evaluate(two_dimensional_ssms.ACT, 'ACT',
                                            {'toreturn':'dataframe'},
                                            eval_efficiency, results, path_save)
        
        results, eval_efficiency = evaluate(get_EI, 'EI', # D_safe is the buffer and can be adjusted
                                            {'toreturn':'dataframe', 'D_safe':0.},
                                            eval_efficiency, results, path_save)

        results = results[['event_id','target_id','time','TAdv','TTC2D','ACT','EI']]
        results.to_hdf(path_save + f'TAdv_TTC2D_ACT_EI.h5', key='data', mode='w')

    # SSSE models in this study
    model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
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
        print(f'--- Evaluating {model_name} ---')

        if os.path.exists(path_save + f'{model_name}.h5'):
            print(f'The events has been evaluated by {model_name}.')
            continue

        # Define scaler and one-hot encoder for normalisation
        current_scaler = get_scaler(dataset, path_prepared, feature=encoder_selection[0])
        if 'profiles' in encoder_selection:
            profiles_scaler = get_scaler(dataset, path_prepared, feature='profiles')
        if 'environment' in encoder_selection:
            environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
            one_hot_encoder = create_categorical_encoder(events, environment_feature_names)

        # Define and load trained model
        model = define_model(device, path_prepared, dataset, encoder_selection, cross_attention, pretrained_encoder)

        states = []
        if encoder_selection[0]=='current':
            states.append(np.concatenate([
                current_scaler.transform(current_features[:,:-2]),
                current_features[:,[-1]]
            ]))
        if encoder_selection[0]=='current+acc':
            states.append(np.concatenate([
                current_scaler.transform(current_features[:,:-1]),
                current_features[:,[-1]]
            ]))
        if 'environment' in encoder_selection:
            environment_features = events.loc[event_id_list[:,0], environment_feature_names].fillna('Unknown')
            environment_features = one_hot_encoder.transform(environment_features.values)
            states.append(environment_features)
        if 'profiles' in encoder_selection:
            states.append(profiles_scaler.transform(profiles_features.reshape(-1, 4)).reshape(profiles_features.shape))

        if len(states) == 1: # only current features
            states = [states[0], spacing_list]
        else:
            states = [tuple(states), spacing_list]

        time_start = systime.time()
        mu, sigma, max_intensity = SSSE(states, model, device)
        time_end = systime.time()

        results = pd.DataFrame(event_id_list, columns=['event_id','target_id','time'])
        results[['event_id','target_id']] = results[['event_id','target_id']].astype(int)
        results['proximity'] = spacing_list
        results['mu'] = mu
        results['sigma'] = sigma
        results['intensity'] = max_intensity
        results.to_hdf(path_save + f'{model_name}.h5', key='data', mode='w')
        eval_efficiency.loc[len(eval_efficiency)] = [model_name, time_end-time_start, results['target_id'].nunique(), len(results)]
        eval_efficiency.to_csv(path_save + 'EvaluationEfficiency.csv', index=False)

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
