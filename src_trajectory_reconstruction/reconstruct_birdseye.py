'''
This script reconstructs the bird's eye view of events.
'''

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from reconstruction_utils.utils_ego_sur import *

path_raw = './RawData/'
path_raw_honda = './RawData/HondaDataSupport/'
path_raw_das = './RawData/DriverAssistanceSystems/'
path_processed = './ProcessedData/SHRP2/'


# Load metadata and event information
meta_both = pd.read_csv(path_processed + 'metadata_birdseye.csv')
events = pd.read_csv(path_raw_honda + 'InsightTables_csv/Event_Table.csv')
meta_both = meta_both.set_index('event_id')
events = events.set_index('eventID')


def record_invalid(meta_both, event_id, ego, surrounding, note):
    meta_both.loc[event_id, 'ego_reconstructed'] = ego
    meta_both.loc[event_id, 'surrounding_reconstructed'] = surrounding
    meta_both.loc[event_id, 'note'] = note
    return meta_both


def update_chunk_save(data_save, chunk_id):
    data_save[np.where(data_save)[0][chunk_id]+1] = True
    data_save[np.where(data_save)[0][chunk_id]] = False
    print(f'\n Updated chunk saving points: {np.where(data_save)[0]}')
    return data_save


def act_before_continue(meta_both, event_id, ego, surrounding, note, data_save, separate_chunk, save_checkpoint, chunk_id):
    meta_both = record_invalid(meta_both, event_id, ego, surrounding, note)
    if separate_chunk and save_checkpoint:
        data_save = update_chunk_save(data_save, chunk_id)
    return meta_both, data_save


# Read ekf parameters
ekf_params = pd.read_csv(path_processed + 'ekf_parameters.csv')

ego_params = dict()
sur_params = dict()
for param in ekf_params.columns:
    if 'ego_' in param:
        ego_params[param.split('ego_')[1]] = ekf_params[param].values[0]
    elif 'sur_' in param:
        sur_params[param.split('sur_')[1]] = ekf_params[param].values[0]
print('Ego parameters: ', ego_params, 'Average error: ', -ekf_params['score_ego'].values[0])
print('Surrounding parameters: ', sur_params, 'Average error: ', -ekf_params['score_sur'].values[0])


# For each event category, reconstruct bird's eye view of events
target_id = 0 # Initialize target_id for surrounding vehicles detected by radar
for event_cat in meta_both['event_category'].value_counts().index.values[::-1]:
    path_events = path_processed + event_cat + '/'
    os.makedirs(path_events, exist_ok=True)

    ## for unsafe events
    if os.path.exists(path_events + 'Ego_birdseye.h5') and os.path.exists(path_events + 'Surrounding_birdseye.h5'):
        print(f"Reconstructed bird\'s eye view of {event_cat} events already exist.")
        data_sur = pd.read_hdf(path_events + 'Surrounding_birdseye.h5', key='data')
        target_id = data_sur['target_id'].max() + 1
        continue
    ## for safe baselines
    if os.path.exists(path_events + 'Ego_birdseye_4.h5') and os.path.exists(path_events + 'Surrounding_birdseye_4.h5'):
        print(f"Reconstructed bird\'s eye view of {event_cat} events already exist.")
        continue

    print(f"Reconstructing bird\'s eye view of {event_cat} events...")
    data_ego = []
    data_sur = []
    event_ids = meta_both[(meta_both['event_category']==event_cat)&
                          (~meta_both['file2use'].isna())].index.sort_values().values
    data_concat = np.zeros(len(event_ids)).astype(bool)
    data_concat[100::100] = True
    data_concat[-1] = False

    data_save = np.zeros(len(event_ids)).astype(bool)
    separate_chunk = len(event_ids)>6600
    chunk_id = 0
    if separate_chunk:
        data_save[6600::6600] = True
        data_save[-1] = False # do not save the last chunk as it will be saved outside the loop
        print(f'Chunk saving points: {np.where(data_save)[0]}')
    else:
        pdf = PdfPages(path_events + 'plots_ego_ekf.pdf')

    for count, event_id, concat_checkpoint, save_checkpoint in tqdm(zip(range(len(event_ids)), event_ids, data_concat, data_save), total=len(event_ids)):
        if separate_chunk:
            file_exist = os.path.exists(path_events + 'Ego_birdseye_' + str(chunk_id) + '.h5') and os.path.exists(path_events + 'Surrounding_birdseye_' + str(chunk_id) + '.h5')
            if not file_exist:
                if count==0:
                    pdf = PdfPages(path_events + 'plots_ego_ekf_0.pdf') # save per 6600 reconstructed trajectories to a pdf file
                elif count==(np.where(data_save)[0][chunk_id-1]+1):
                    pdf = PdfPages(path_events + 'plots_ego_ekf_' + str(chunk_id) + '.pdf')
        if separate_chunk and file_exist:
            if save_checkpoint:
                print(f' Current chunk {chunk_id} already exists, continue ...')
                data_sur = pd.read_hdf(path_events + 'Surrounding_birdseye_' + str(chunk_id) + '.h5', key='data')
                target_id = data_sur['target_id'].max() + 1
                chunk_id += 1
                data_sur = []
            continue

        ## read time series data
        try:
            sample = pd.read_csv(meta_both.loc[event_id]['file_dir'] + meta_both.loc[event_id]['file2use'], on_bad_lines='warn')
        except:
            meta_both, data_save = act_before_continue(meta_both, event_id, 0, 0, 'file reading error', 
                                                    data_save, separate_chunk, save_checkpoint, chunk_id)
            continue
        
        ## check if ego speed is all nan
        if sample['vtti.speed_network'].isna().all():
            meta_both, data_save = act_before_continue(meta_both, event_id, 0, 0, 'all nan ego speed',
                                                    data_save, separate_chunk, save_checkpoint, chunk_id)
            continue

        if sample['vtti.accel_x'].isna().all() or sample['vtti.accel_y'].isna().all():
            meta_both, data_save = act_before_continue(meta_both, event_id, 0, 0, 'all nan acceleration',
                                                    data_save, separate_chunk, save_checkpoint, chunk_id)
            continue

        if sample['vtti.gyro_z'].isna().all():
            meta_both, data_save = act_before_continue(meta_both, event_id, 0, 0, 'all nan yaw rate',
                                                    data_save, separate_chunk, save_checkpoint, chunk_id)
            continue

        ## create dataframe (note: SHRP2 excluded rearward vehicles)
        df_ego, df_forward, target_id, reconnected_ids = create_dataframe(sample, event_id, target_id)
        if np.all(df_ego['speed_comp']<=1e-6):
            meta_both, data_save = act_before_continue(meta_both, event_id, 0, 0, 'all zero ego speed',
                                                    data_save, separate_chunk, save_checkpoint, chunk_id)
            continue
        
        ## check if some targets are reconnected
        if len(reconnected_ids)>0:
            meta_both.loc[event_id, 'note'] = f'Targets {reconnected_ids.keys()} are reconnected to {reconnected_ids.values()}'

        ## reconstruct ego trajectory and make comparison plots
        df_ego, valid_ego, pdf = process_ego(df_ego, event_id, pdf, ego_params)
        if not valid_ego:
            meta_both, data_save = act_before_continue(meta_both, event_id, 0, 0, 'start&end speed not available',
                                                    data_save, separate_chunk, save_checkpoint, chunk_id)
            continue

        ## reconstruct surrounding vehicle trajectory if there are useful surrounding vehicles
        if len(df_forward)>0:
            ego_length = meta_both.loc[event_id]['ego_length']
            if np.isnan(ego_length):
                df_sur = pd.DataFrame()
                meta_both, data_save = act_before_continue(meta_both, event_id, 1, 0, 'no ego width or length',
                                                    data_save, separate_chunk, save_checkpoint, chunk_id)
            else:
                df_forward = process_surrounding(df_ego, df_forward, ego_length, sur_params)
                if len(df_forward)>0:
                    df_sur = df_forward.copy()
                else:
                    df_sur = pd.DataFrame()
                    meta_both = record_invalid(meta_both, event_id, 1, 0, 'no long enough surrounding trajectories')
        else:
            df_sur = pd.DataFrame()
            meta_both = record_invalid(meta_both, event_id, 1, 0, 'no available surrounding trajectories')
        
        ## mark segments covering the event
        timestamp_start = events.loc[event_id]['eventStart']
        timestamp_end = events.loc[event_id]['eventEnd']
        if timestamp_end=='(null)':
            df_ego['event'] = 0
        else:
            timestamp_start = int(timestamp_start)
            timestamp_end = int(timestamp_end)
            df_ego.loc[(df_ego['timestamp']>=timestamp_start)&(df_ego['timestamp']<=timestamp_end), 'event'] = 1
            df_ego.loc[df_ego['event'].isna(), 'event'] = 0

        ## append dataframes
        if concat_checkpoint:
            data_ego = pd.concat(data_ego)
            data_sur = pd.concat(data_sur)
            data_ego = [data_ego]
            data_sur = [data_sur]
        data_ego.append(df_ego)
        data_sur.append(df_sur)

        ## save dataframe every 6600 events to avoid memory error
        if save_checkpoint:
            data_ego = pd.concat(data_ego).reset_index(drop=True).infer_objects()
            data_sur = pd.concat(data_sur).reset_index(drop=True).infer_objects()
            data_ego[['event_id','timestamp','event']] = data_ego[['event_id','timestamp','event']].astype(int)
            data_sur[['event_id','target_id','timestamp']] = data_sur[['event_id','target_id','timestamp']].astype(int)
            target_id = data_sur['target_id'].max() + 1

            data_ego.to_hdf(path_events + 'Ego_birdseye_' + str(chunk_id) + '.h5', key='data')
            data_sur.to_hdf(path_events + 'Surrounding_birdseye_' + str(chunk_id) + '.h5', key='data')
            print(f' Current chunk: {chunk_id} ...')
            chunk_id += 1
            data_ego = []
            data_sur = []
            
            pdf.close() # close the pdf file and create a new one for the next chunk
            pdf = PdfPages(path_events + 'plots_ego_ekf_' + str(chunk_id) + '.pdf')

    pdf.close() # close the pdf file when the loop ends

    # save the last dataframes
    data_ego = pd.concat(data_ego).reset_index(drop=True).infer_objects()
    data_sur = pd.concat(data_sur).reset_index(drop=True).infer_objects()
    data_ego[['event_id','timestamp','event']] = data_ego[['event_id','timestamp','event']].astype(int)
    data_sur[['event_id','target_id','timestamp']] = data_sur[['event_id','target_id','timestamp']].astype(int)
    target_id = data_sur['target_id'].max() + 1

    if separate_chunk:
        data_ego.to_hdf(path_events + 'Ego_birdseye_' + str(chunk_id) + '.h5', key='data')
        data_sur.to_hdf(path_events + 'Surrounding_birdseye_' + str(chunk_id) + '.h5', key='data')
    else:
        data_ego.to_hdf(path_events + 'Ego_birdseye.h5', key='data')
        data_sur.to_hdf(path_events + 'Surrounding_birdseye.h5', key='data')

    meta_both.loc[data_ego['event_id'].unique(), 'ego_reconstructed'] = 1
    meta_both.loc[data_sur['event_id'].unique(), 'surrounding_reconstructed'] = 1
    meta_both.to_csv(path_processed + 'metadata_birdseye.csv', index=True)

meta_both.loc[meta_both['ego_reconstructed'].isna(), 'ego_reconstructed'] = 0
meta_both.loc[meta_both['surrounding_reconstructed'].isna(), 'surrounding_reconstructed'] = 0
meta_both.to_csv(path_processed + 'metadata_birdseye.csv', index=True)
print('Event category events double check: ', meta_both['event_category'].value_counts())
