'''
This script is to make votes using various SSMs to select the conflicting target for each event in the SHRP2 dataset.
'''

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as systime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_safety_evaluation.validation_utils.utils_evaluation import read_events


def main(path_result):
    initial_time = systime.time()
    print('---- available cpus:', os.cpu_count(), '----')

    # Read data for all event categories
    path_events = path_result + 'EventData/'
    path_results = path_result + 'ConflictingTarget/'
    os.makedirs(path_results, exist_ok=True)
    event_meta = read_events(path_events, meta_only=True)
    event_meta = event_meta[['duration_enough','event_category','conflict']]

    # Record the identified target by all used SSMs under corresponding optimal thresholds
    warning_timeliness = pd.read_hdf(path_result + 'Analyses/OptimalWarningEvaluation.h5', key='results')
    models = warning_timeliness['model'].unique()
    models = [model for model in models if 'ArgoverseAV' not in model]
    for model in models:
        warning_model = warning_timeliness[warning_timeliness['model']==model]
        event_meta.loc[warning_model['event_id'].values, model] = warning_model['target_id'].values
    event_meta[models] = event_meta[models].fillna(-1).astype(int)

    # Seeing each model makes a vote, select the target with the most votes 
    # and less than 1/3 of the total votes against (considering NaNs as abstentions)
    for event_id in tqdm(event_meta.index, desc='Vote for conflicting target', ascii=True, miniters=100):
        candidates = event_meta.loc[event_id][models].values
        candidates, votes = np.unique(candidates, return_counts=True)
        if np.any(candidates<0):
            abstentions = votes[candidates<0][0]
            votes = votes[candidates>=0]
            candidates = candidates[candidates>=0]
        else:
            abstentions = 0
        if len(candidates) < 1:
            event_meta.loc[event_id, 'target_id'] = -1
            event_meta.loc[event_id, 'target_note'] = 'No target is identified by any SSMs'
            continue
        if votes.max() < len(models)/3:
            event_meta.loc[event_id, 'target_id'] = -1
            event_meta.loc[event_id, 'target_note'] = 'The most votes are less than 1/3 of total'
            continue
        most_voted = candidates[votes.argmax()]
        if votes.sum()-votes.max() >= len(models)/3:
            event_meta.loc[event_id, 'target_id'] = -1
            event_meta.loc[event_id, 'target_note'] = 'More than 1/3 votes are not for the most voted target'
            continue
        event_meta.loc[event_id, 'target_id'] = most_voted
        event_meta.loc[event_id, 'target_note'] = f'For: {votes.max()}, against: {votes.sum()-votes.max()}, abstentions: {abstentions}'

    event_meta['target_id'] = event_meta['target_id'].astype(int)
    event_meta['target_note'] = event_meta['target_note'].astype(str)
    event_meta.to_csv(path_results + 'Voted_conflicting_target.csv', index=True, header=True)

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    path_result = 'ResultData/'
    main(path_result)
