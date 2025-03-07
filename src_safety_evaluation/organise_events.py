'''
This script organises the metadata, data, and features for event evaluation.
'''

import os
import sys
import time as systime
from tqdm import tqdm
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_safety_evaluation.validation_utils.utils_evaluation import set_veh_dimensions
from src_safety_evaluation.validation_utils.utils_features import read_data, get_context_representations

manual_seed = 131
path_processed = 'ProcessedData/SHRP2/'
path_prepared = 'PreparedData/'
path_result = 'ResultData/'
os.makedirs(path_result, exist_ok=True)


def main(meta_both, events):
    initial_time = systime.time()    
    meta_both = meta_both[(meta_both['event_category']!='SafeBaseline')&
                          (meta_both['ego_reconstructed'].astype(bool))&
                          (meta_both['surrounding_reconstructed'].astype(bool))]
    
    path_save = path_result + 'EventData/'
    os.makedirs(path_save, exist_ok=True)
    for event_cat in meta_both['event_category'].value_counts().index.values[::-1]:
        event_meta = meta_both[meta_both['event_category']==event_cat].copy()
        os.makedirs(path_save + f'{event_cat}/', exist_ok=True)
        if os.path.exists(path_save + f'{event_cat}/event_data.h5'):
            print(f'---- {event_cat} data have been organised ----')
            data = pd.read_hdf(path_save + f'{event_cat}/event_data.h5', key='data')
            event_meta = pd.read_csv(path_save + f'{event_cat}/event_meta.csv').set_index('event_id')
        else:
            data = read_data(event_cat)
            assert data['event_id'].nunique() == len(event_meta)
            data = data.reset_index().set_index('event_id')

            for event_id in tqdm(event_meta.index.values, desc=f'{event_cat} data', ascii=True, dynamic_ncols=False, miniters=100):
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


        # Save event features
        assert np.all(np.isin(data['event_id'].unique(), event_meta.index.values))
        if os.path.exists(path_save + f'{event_cat}/event_features.npz'):
            print(f'---- Event features for {event_cat} have been saved ----')
            event_featurs = np.load(path_result + f'EventData/{event_cat}/event_features.npz')
            profiles_features = event_featurs['profiles']
            current_features = event_featurs['current']
            spacing_list = event_featurs['spacing']
            event_id_list = event_featurs['event_id']
        else:
            avg_width = np.nanmean(event_meta['ego_width'].values)
            avg_length = np.nanmean(event_meta['ego_length'].values)
            veh_dimensions = set_veh_dimensions(event_meta, avg_width, avg_length)

            profiles_features = []
            current_features = []
            spacing_list = []
            event_id_list = []
            target_ids = data[data['event_id'].isin(event_meta[event_meta['duration_enough']].index.values)].index.unique(level='target_id').values
            for target_id in tqdm(target_ids, desc=f'{event_cat} features', position=0, dynamic_ncols=False, ascii=True, miniters=min(len(target_ids)//10, 150)):
                df = data.loc(axis=0)[target_id, :]
                if len(df)<35: # skip if the target was detected for less than 3.5 seconds
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

            assert profiles_features.shape[0] == len(spacing_list) and profiles_features.shape[1] == 25
            np.savez(path_save + f'{event_cat}/event_features.npz', 
                     profiles = profiles_features, 
                     current = current_features, 
                     spacing = spacing_list, 
                     event_id = event_id_list)
            
        # Print variable descriptions
        current_features = pd.DataFrame(current_features, columns=['ego_length','target_length','combined_width',
                                                                   'vy_ego','vx_sur','vy_sur','v_ego2','v_sur2','delta_v2','delta_v',
                                                                   'psi_sur','acc_ego','rho'])
        current_features['s'] = spacing_list
        profiles_features = pd.DataFrame(profiles_features.reshape(-1, 4), columns=['acc_ego','v_ego','vx_sur','vy_sur'])
        print(f'--------------------- Variables in {event_cat} ---------------------')
        print(data.columns.to_list(), '\n')
        print('Current features:\n', current_features.describe().to_string(), '\n')
        print('Profiles features:\n', profiles_features.describe().to_string())
        print('--------------------------------------------------------------------')
        del current_features, profiles_features, spacing_list, event_id_list

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    manual_seed = 131
    np.random.seed(manual_seed)

    # Load metadata and event information
    meta_both = pd.read_csv(path_processed + 'metadata_birdseye.csv')
    meta_both = meta_both.set_index('event_id')
    events = pd.read_csv('RawData/SHRP2/HondaDataSupport/InsightTables_csv/Event_Table.csv').set_index('eventID')
    
    main(meta_both, events)
