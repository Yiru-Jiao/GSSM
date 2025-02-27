'''
This script organises the event data and metadata for event evaluation.
'''

import os
import sys
import time as systime
from tqdm import tqdm
import numpy as np
import pandas as pd
from validation_utils.utils_features import read_data

manual_seed = 131
path_processed = './ProcessedData/SHRP2/'
path_prepared = './PreparedData/'
path_result = './ResultData/'


def main(meta_both, events):
    initial_time = systime.time()    
    meta_both = meta_both[(meta_both['event_category']!='SafeBaseline')&
                          (meta_both['ego_reconstructed'].astype(bool))&
                          (meta_both['surrounding_reconstructed'].astype(bool))]
    
    path_save = path_result + 'EventData/'
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

        for event_id in tqdm(event_meta.index.values, desc=event_cat, ascii=True, dynamic_ncols=False, miniters=100):
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

            # Retrieve event narrative
            event_meta.loc[event_id, 'narrative'] = events.loc[event_id, 'finalNarrative']

            # Annotate duration_enough: at least one target is recorded over 5 seconds
            target_duration = df.groupby('target_id')['time'].count()
            if target_duration.max()>=50:
                event_meta.loc[event_id, 'duration_enough'] = True
            else:
                event_meta.loc[event_id, 'duration_enough'] = False

        # Remove events with missing timestamps
        data = data.reset_index().sort_values(['target_id','time']).set_index(['target_id','time'])
        data = data[data['event_id'].isin(event_meta[event_meta['duration_enough']].index.values)]
        event_ids_to_remove = event_meta[event_meta[['start_timestamp', 'end_timestamp', 'impact_timestamp']].isnull().any(axis=1)].index.values
        data = data[~data['event_id'].isin(event_ids_to_remove)]

        # Save data
        event_meta = event_meta.drop(columns=['time_series_das', 'time_series_honda', 'file_dir', 'file2use', 'ego_reconstructed', 'surrounding_reconstructed', 'note'])
        event_meta.to_csv(path_save + f'{event_cat}/event_meta.csv')
        data.to_hdf(path_save + f'{event_cat}/event_data.h5', key='data', mode='w')
        print(f'Variables in {event_cat}: ', data.describe().to_string())

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
