'''
This script is to make votes using various SSMs to select the conflicting target for each event in the SHRP2 dataset.
'''

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as systime
manual_seed = 131


def main(path_result):
    initial_time = systime.time()
    print('---- available cpus:', os.cpu_count(), '----')

    '''
    Vote for the conflicting targets using all SSM models.
    '''
    # Read meta data for all event categories
    os.makedirs(path_result + 'Conflicts/', exist_ok=True)
    event_meta = pd.read_csv(path_result + 'Analyses/EventMeta.csv', index_col=0)

    if os.path.exists(path_result + 'Conflicts/Voted_conflicting_targets.csv'):
        voted_targets = pd.read_csv(path_result + 'Conflicts/Voted_conflicting_targets.csv', index_col=0)
        print('--- Conflicting targets already voted ---')
    else:
        print('--- Voting conflicting targets ---')
        voted_targets = event_meta[['duration_enough','event_category','conflict']].copy()

        # Record the identified target by all used SSMs
        warning_files = sorted(os.listdir(path_result + 'Analyses/'))
        warning_files = [f for f in warning_files if f.startswith('Warning_') and f.endswith('.h5')]

        models = []
        for warning_file in warning_files:
            model = warning_file.split('Warning_')[1][:-3]
            if 'UCD' in model:
                continue
            if 'mixed' in model:
                if 'ArgoverseHV' in model:
                    if '0.1' not in model:
                        continue
                elif 'highD' in model:
                    if '1.0' not in model:
                        continue
            models.append(model)
            warning = pd.read_hdf(path_result+'Analyses/'+warning_file, key='results')
            warning = warning.groupby('event_id')['target_id'].first()
            voted_targets.loc[warning.index, model] = warning.values
        voted_targets[models] = voted_targets[models].fillna(-1).astype(int)
        print(f'Models to use ({len(models)}):\n', models)

        # Seeing each model makes a vote, select the target with the most votes 
        # and less than 1/3 of the total votes against (considering NaNs as abstentions)
        for event_id in tqdm(voted_targets.index, desc='Vote for conflicting target', ascii=True, miniters=100):
            candidates = voted_targets.loc[event_id][models].values
            candidates, votes = np.unique(candidates, return_counts=True)
            if np.any(candidates<0):
                abstentions = votes[candidates<0][0]
                votes = votes[candidates>=0]
                candidates = candidates[candidates>=0]
            else:
                abstentions = 0
            if len(candidates) < 1:
                voted_targets.loc[event_id, 'target_id'] = -1
                voted_targets.loc[event_id, 'target_note'] = 'No target is identified by any SSMs'
                continue
            if votes.max() < len(models)/3:
                voted_targets.loc[event_id, 'target_id'] = -1
                voted_targets.loc[event_id, 'target_note'] = 'The most votes are less than 1/3 of total'
                continue
            most_voted = candidates[votes.argmax()]
            if votes.sum()-votes.max() >= len(models)/3:
                voted_targets.loc[event_id, 'target_id'] = -1
                voted_targets.loc[event_id, 'target_note'] = 'More than 1/3 votes are not for the most voted target'
                continue
            voted_targets.loc[event_id, 'target_id'] = most_voted
            voted_targets.loc[event_id, 'target_note'] = f'For: {votes.max()}, against: {votes.sum()-votes.max()}, abstentions: {abstentions}'

        voted_targets['target_id'] = voted_targets['target_id'].astype(int)
        voted_targets['target_note'] = voted_targets['target_note'].astype(str)
        voted_targets.to_csv(path_result + 'Conflicts/Voted_conflicting_targets.csv', index=True, header=True)

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(manual_seed)
    path_result = 'ResultData/'
    main(path_result)
