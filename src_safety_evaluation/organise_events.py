'''
'''

import os
import sys
import time as systime
from tqdm import tqdm
import numpy as np
import pandas as pd
from validation_utils.utils_features import read_data


manual_seed = 131
path_processed = './ProcessedData/'
path_prepared = './PreparedData/'
path_result = './ResultData/'



def main(meta_both, events):
    initial_time = systime.time()    
    meta_both = meta_both[(meta_both['event_category']!='SafeBaseline')&
                          (meta_both['ego_reconstructed'].astype(bool))&
                          (meta_both['surrounding_reconstructed'].astype(bool))]
    
    path_save = path_result + 'EventEvaluation/'
    os.makedirs(path_save, exist_ok=True)
    for event_cat in meta_both['event_category'].value_counts().index.values[::-1]:
        event_meta = meta_both[meta_both['event_category']==event_cat].copy()
        print(f'--- Organising data and meta for {event_cat} ---')
        os.makedirs(path_save + f'{event_cat}/', exist_ok=True)
        if os.path.exists(path_save + f'{event_cat}/event_data.h5'):
            print(f'--- {event_cat} already organised ---')
            continue

        data = read_data(event_cat)
        assert data['event_id'].nunique() == len(event_meta)
        data = data.reset_index().set_index('event_id')

        for event_id in tqdm(event_meta.index.values, desc='Event'):
            df = data.loc[event_id]
            # Annotate key timestamps
            raw_vars = ['eventStart', 'eventEnd', 'impactProximity', 'subjectReactionStart']
            new_vars = ['start', 'end', 'impact', 'reaction']
            for raw_name, new_name in zip(raw_vars, new_vars):
                value = str(events.loc[event_id, raw_name])
                if value=='(null)' or value=='-99' or value=='-1':
                    value = np.nan
                event_meta.loc[event_id, f'{new_name}_timestamp'] = float(value)

            # Annotate event severity
            '''
            For each event, we mark the severiest level if more than one are present
            Crash: 3
            NearCrash: 2
            CrashRelevant: 1
            Not applicable: 0
            The target is not considered if it is a following vehicle, as the data does not provide rear radar information
            '''
            if event_cat=='Crash':
                event_meta.loc[event_id, 'severity_first'] = 3
                event_meta.loc[event_id, 'severity_second'] = 0
            elif event_cat=='NearCrash':
                event_meta.loc[event_id, 'severity_first'] = 2
                event_meta.loc[event_id, 'severity_second'] = 0
            elif 'Secondary' in event_cat:
                event_meta.loc[event_id, 'severity_first'] = 0
                if event_cat=='SecondaryCrash':
                    event_meta.loc[event_id, 'severity_second'] = 3
                elif event_cat=='SecondaryNearCrash':
                    event_meta.loc[event_id, 'severity_second'] = 2
            elif '-' in event_cat:
                first, second = event_cat.split('-')
                event_meta.loc[event_id, 'severity_first'] = 3 if first=='Crash' else 2 if first=='NearCrash' else 1
                event_meta.loc[event_id, 'severity_second'] = 3 if second=='Crash' else 2 if second=='NearCrash' else 1
            if event_meta.loc[event_id, 'first'] in ['animal', 'following', 'obstacle', 'single']:
                event_meta.loc[event_id, 'severity_first'] = 0
            if event_meta.loc[event_id, 'second'] in ['animal', 'following', 'obstacle', 'single']:
                event_meta.loc[event_id, 'severity_second'] = 0

            # Retrieve event narrative
            event_meta.loc[event_id, 'narrative'] = events.loc[event_id, 'finalNarrative']

            # Annotate applicability
            '''
            An event is applicable if
            - duration_enough: at least one target is recorded over 2.5 seconds
            - danger_present: at least one target with long enough duration presents during [impact_timestamp-0.5 s, impact_timestamp+0.5 s]
            - reaction_covered: at least one target presented in danger also presents 1 sec earlier than reaction_timestamp
            - safety_danger:
                * danger: at most 3 seconds before impact_timestamp and after start_timestamp
                * safety: first 3 seconds before start_timestamp with 
                          1) no hard braking, i.e., acceleration > -1.5 m/s^2 in the 3 seconds
                          2) not in congestion, i.e., both ego and target speed > 3 m/s at the initial moment
            '''
            target_time = df.groupby('target_id')['time'].agg(['min', 'max', 'count'])
            if target_time['count'].max()>=25:
                event_meta.loc[event_id, 'duration_enough'] = True
            else:
                event_meta.loc[event_id, 'duration_enough'] = False

            target_present = target_time[(target_time['count']>=25)&
                                         (target_time['min']<=event_meta.loc[event_id, 'impact_timestamp']/1000-0.5)&
                                         (target_time['max']>=event_meta.loc[event_id, 'impact_timestamp']/1000+0.5)]
            if len(target_present)>0:
                event_meta.loc[event_id, 'danger_present'] = True
            else:
                event_meta.loc[event_id, 'danger_present'] = False

            target_covered = target_present[target_present['min']<=event_meta.loc[event_id, 'reaction_timestamp']/1000-1.]
            if len(target_covered)>0:
                event_meta.loc[event_id, 'reaction_covered'] = True
            else:
                event_meta.loc[event_id, 'reaction_covered'] = False

            targets = df[df['target_id'].isin(target_present[target_present['min']<=event_meta.loc[event_id, 'start_timestamp']/1000-3.].index.values)]
            if len(targets)>0:
                targets_first3s = targets[targets['time']<=targets.groupby('target_id')['time'].transform('min')+3.]
                no_hard_braking = targets_first3s.groupby('target_id')['acc_ego'].min()>-1.5
                initial_speeds = targets_first3s.groupby('target_id')[['v_ego', 'v_sur']].first()
                not_in_congestion = (initial_speeds['v_ego']>3.)&(initial_speeds['v_sur']>3.)
                safe_targets = no_hard_braking[no_hard_braking&not_in_congestion].index.values
                safe_targets = targets_first3s[targets_first3s['target_id'].isin(safe_targets)]
                if len(safe_targets)>0:
                    event_meta.loc[event_id, 'safety_danger'] = True
                else:
                    event_meta.loc[event_id, 'safety_danger'] = False
            else:
                event_meta.loc[event_id, 'safety_danger'] = False

            if event_meta.loc[event_id, 'safety_danger']:
                danger_start = max(event_meta.loc[event_id, 'impact_timestamp']-3000, event_meta.loc[event_id, 'start_timestamp'])
                danger_end = min(event_meta.loc[event_id, 'impact_timestamp']+500, event_meta.loc[event_id, 'end_timestamp'])
                # safety start and end will depend on the target being evaluated
                event_meta.loc[event_id, 'danger_start'] = danger_start
                event_meta.loc[event_id, 'danger_end'] = danger_end

        # Remove events with missing timestamps
        data = data.reset_index().sort_values(['target_id','time']).set_index(['target_id','time'])
        event_ids_to_remove = event_meta[event_meta[['start_timestamp', 'end_timestamp', 'impact_timestamp']].isnull().any(axis=1)].index.values
        data = data[~data['event_id'].isin(event_ids_to_remove)]

        # Save data
        event_meta = event_meta.drop(columns=['time_series_das', 'time_series_honda', 'file_dir', 'file2use', 'ego_reconstructed', 'surrounding_reconstructed', 'note'])
        event_meta.to_csv(path_save + f'{event_cat}/event_meta.csv')
        data.to_hdf(path_save + f'{event_cat}/event_data.h5', key='data', mode='w')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    manual_seed = 131
    np.random.seed(manual_seed)

    # Load metadata and event information
    meta_both = pd.read_csv(path_processed + 'metadata_birdseye.csv')
    meta_both = meta_both.set_index('event_id')
    events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_Table.csv').set_index('eventID')
    
    main(meta_both, events)
