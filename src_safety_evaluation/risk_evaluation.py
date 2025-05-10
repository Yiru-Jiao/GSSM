'''
This script is to make votes using various SSMs to select the conflicting target for each event in the SHRP2 dataset.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as systime
from joblib import Parallel, delayed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_safety_evaluation.validation_utils.utils_evaluation import read_events, read_evaluation, evaluate, optimize_threshold, issue_warning
manual_seed = 131


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reversed_list', type=int, default=0, help='Whether to reverse the model list (defaults to False), useful for running parallel jobs')
    args = parser.parse_args()
    args.reversed_list = bool(args.reversed_list)
    return args


def get_model_fig(model_name):
    dataset_name = model_name.split('_current')[0]

    if 'acc' in model_name:
        features = model_name.split('current+acc')[1]
        features = features.split('_')
        features = ['current+acc'] + features
    else:
        features = model_name.split('current')[1]
        features = features.split('_')
        features = ['current'] + features
    features = [f for f in features if f !='']
    encoder_name = '_'.join(features)

    return dataset_name, encoder_name


def fill_na_warning(results):
    results.loc[results['danger_recorded'].isna(), 'danger_recorded'] = False
    results.loc[results['safety_recorded'].isna(), 'safety_recorded'] = False
    results[['danger_recorded', 'safety_recorded']] = results[['danger_recorded', 'safety_recorded']].astype(bool)
    results[['safe_target_ids', 'indicator', 'model']] = results[['safe_target_ids', 'indicator', 'model']].astype(str)
    return results


def main(args, path_result, path_prepared):
    initial_time = systime.time()
    print('---- available cpus:', os.cpu_count(), '----')

    if os.path.exists(path_result + 'Conflicts/Voted_conflicting_targets.csv'):
        voted_targets = pd.read_csv(path_result + 'Conflicts/Voted_conflicting_targets.csv', index_col=0)
        print('--- Conflicting targets already voted ---')
    else:
        print('--- Get the voted conflicting targets first! ---')
        sys.exit(0)
    
    '''
    Evaluate the performance of models with the determined conflicting targets.
    '''
    path_events = path_result + 'EventData/'
    path_eval = path_result + 'EventEvaluation/'
    os.makedirs(path_result + 'Conflicts/Results/', exist_ok=True)
    _, event_data = read_events(path_events)
    event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)
    voted_targets = voted_targets[voted_targets['target_id']>=0]

    # 2D SSMs and UCD
    for indicator in ['TAdv', 'TTC2D', 'ACT', 'EI', 'UCD']:
        if args.reversed_list:
            continue
        if indicator == 'TAdv':
            thresholds = np.unique(np.round(np.arange(0,1.75,0.0115)**7,2))
        elif indicator in ['TTC2D', 'ACT']:
            thresholds = np.unique(np.round(np.arange(0,1.94,0.0135)**7,2))
        elif indicator == 'EI':
            thresholds = np.round((8**np.arange(0,2.31,0.0265)-1)/50, 2)
            thresholds = np.unique(np.sort(np.concatenate([thresholds, -thresholds[::2]*3])))
        elif indicator == 'UCD':
            thresholds = np.unique(np.round(10**np.arange(0,5.95,0.055))-2)
        
        if os.path.exists(path_result + f'Conflicts/Results/RiskEval_{indicator}.h5'):
            print(f'--- Evaluation with {indicator} already completed ---')
        else:
            print(f'--- Evaluating with {indicator} ---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation(indicator, path_eval)
            progress_bar = tqdm(thresholds, desc=indicator, ascii=True, dynamic_ncols=False, miniters=10)
            records = Parallel(n_jobs=-1)(delayed(evaluate)(indicator, threshold, safety_evaluation, event_data, event_meta, voted_targets) for threshold in progress_bar)
            records = pd.concat(records).reset_index()
            records['indicator'] = indicator
            records['model'] = indicator
            records = fill_na_warning(records)
            records.to_hdf(path_result + f'Conflicts/Results/RiskEval_{indicator}.h5', key='results', mode='w')
            progress_bar.close()
            print(f'{indicator} time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    # GSSM
    model_evaluation = pd.read_csv(path_prepared + 'PosteriorInference/evaluation.csv')
    dataset_name_list = model_evaluation['dataset'].values
    encoder_name_list = model_evaluation['encoder_selection'].values
    mixrate_list = model_evaluation['mixrate'].values
    if args.reversed_list:
        dataset_name_list = dataset_name_list[::-1]
        encoder_name_list = encoder_name_list[::-1]
        mixrate_list = mixrate_list[::-1]

    gssm_thresholds = np.unique(np.round(np.arange(0,6,0.06)-0.06,2))
    for dataset_name, encoder_name, mixrate in zip(dataset_name_list, encoder_name_list, mixrate_list):
        if np.isnan(mixrate):
            model_name = f'{dataset_name}_{encoder_name}'
        else:
            model_name = f'{dataset_name}_{encoder_name}_mixed{mixrate}'
            if 'ArgoverseHV' in model_name:
                if '0.1' not in model_name:
                    continue
            elif 'highD' in model_name:
                if '1.0' not in model_name:
                    continue
        if os.path.exists(path_result + f'Conflicts/Results/RiskEval_{model_name}.h5'):
            print('--- Evaluation with', model_name, 'already completed ---')
        else:
            print('--- Evaluating with', model_name, '---')
            sub_initial_time = systime.time()
            safety_evaluation = read_evaluation('GSSM', path_eval, model_name)
            progress_bar = tqdm(gssm_thresholds, desc=model_name, ascii=True, dynamic_ncols=False, miniters=10)
            gssm_records = Parallel(n_jobs=-1)(delayed(evaluate)('GSSM', threshold, safety_evaluation, event_data, event_meta, voted_targets) for threshold in progress_bar)
            gssm_records = pd.concat(gssm_records).reset_index()
            gssm_records['indicator'] = 'GSSM'
            gssm_records['model'] = model_name
            gssm_records = fill_na_warning(gssm_records)
            gssm_records.to_hdf(path_result + f'Conflicts/Results/RiskEval_{model_name}.h5', key='results', mode='w')
            progress_bar.close()
            print(model_name, 'time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - sub_initial_time)))

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(manual_seed)
    path_prepared = 'PreparedData/'
    path_result = 'ResultData/'
    args = parse_args()
    main(args, path_result, path_prepared)
