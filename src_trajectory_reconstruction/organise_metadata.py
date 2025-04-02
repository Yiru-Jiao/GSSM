'''
This script organizes metadata from Honda Data Support and Driving Assistance Systems, 
and categorizes events based on severity and nature of conflicts.
'''

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

path_raw = './RawData/SHRP2/'
path_raw_honda = './RawData/SHRP2/HondaDataSupport/'
path_raw_das = './RawData/SHRP2/DriverAssistanceSystems/'
path_processed = './ProcessedData/SHRP2/'
os.makedirs(path_raw + 'FileToUse/', exist_ok=True)


# Set of Driving Assistance Systems
if not os.path.exists(path_raw + 'FileToUse/metadata_timeseries_video_DriverAssistanceSystems.csv'):
    path_timeseries = path_raw_das + 'Videos_And_TimeSeriesData/'
    print('Scaning files in Videos_And_TimeSeriesData/ ...')
    filename_list = os.listdir(path_timeseries)

    summary_df = pd.DataFrame([], columns=['event_id', 'time_series', 'video_front', 'video_rear', 'video_hands'])
    id_count = 0
    for filename in tqdm(filename_list, desc='Processing'):
        if filename.endswith('.csv'):
            event_id = int(filename.split('.')[0].split('_')[-1])
            columns = ['event_id', 'time_series']
            values = [event_id, filename]
        elif filename.endswith('.avi') or filename.endswith('.mp4'):
            event_id = int(filename.split('Index_')[-1].split('_')[0])
            columns = ['event_id']
            values = [event_id]
            if 'Front' in filename:
                columns.append('video_front')
                values.append(filename)
            if 'Rear' in filename:
                columns.append('video_rear')
                values.append(filename)
            if 'Hands' in filename:
                columns.append('video_hands')
                values.append(filename)
        if event_id in summary_df['event_id'].values:
            summary_df.loc[summary_df['event_id']==event_id, columns] = values
        else:
            id_count += 1
            summary_df.loc[id_count, columns] = values
    summary_df.to_csv(path_raw + 'FileToUse/metadata_timeseries_video_DriverAssistanceSystems.csv', index=False)


# Set of Honda Data Support
if not os.path.exists(path_raw + 'FileToUse/metadata_timeseries_video_HondaDataSupport.csv'):
    summary_df = pd.DataFrame([], columns=['event_id', 'time_series', 'video_front', 'video_rear', 'video_hands'])
    id_count = 0

    for folder in ['Time Series Export/', 'Export_Video_Redacted/Deidentified Baseline Video/', 'Export_Video_Redacted/Rear Video Near Crash DI/']:
        path_timeseries = path_raw_honda + folder
        print(f'Scaning files in {folder} ...')
        filename_list = os.listdir(path_timeseries)

        for filename in tqdm(filename_list, desc='Processing'):
            if 'Fast' in filename:
                continue
            if filename.endswith('.csv'):
                event_id = int(filename.split('.')[0].split('_')[-1])
                columns = ['event_id', 'time_series']
                values = [event_id, filename]
            elif filename.endswith('.avi') or filename.endswith('.mp4'):
                event_id = int(filename.split('ID_')[-1].split('_')[0])
                columns = ['event_id']
                values = [event_id]
                if 'Front' in filename:
                    columns.append('video_front')
                    values.append(folder+filename)
                elif 'Rear' in filename:
                    columns.append('video_rear')
                    values.append(folder+filename)
                elif 'Hands' in filename:
                    columns.append('video_hands')
                    values.append(folder+filename)
            if event_id in summary_df['event_id'].values:
                summary_df.loc[summary_df['event_id']==event_id, columns] = values
            else:
                id_count += 1
                summary_df.loc[id_count, columns] = values
    summary_df.to_csv(path_raw + 'FileToUse/metadata_timeseries_video_HondaDataSupport.csv', index=False)


# Define vehicle dimensions, the values are estimated by Microsoft Copilot
def define_vehicle_dimension():
    vehicle_dimension = {'CAR': [1.75, 4.5], # w: 1.65~1.9m, l: 4.4~4.8m
                        'PICKUP_TRUCK': [2., 6.], # w: 1.9~2.1m, l: 5.4~6.4m
                        'SUV_CROSSOVER': [1.8, 4.6], # w: 1.76~1.9m, l: 4.5~4.7m
                        'VAN_MINIVAN': [1.9, 5.2], # w: 1.85~2m, l: 5.1~5.2m
                        '(null)': [np.nan, np.nan],
                        'Ambulance': [2.4, 6.5], # w: 2.3~2.5m, l: 6~7m
                        'Animal': [0., 0.],
                        'Automobile': [1.75, 4.5],
                        'Cyclist': [0.5, 1.5],
                        'Fire truck/car': [3., 12.], # w: ~3m, l: ~12m
                        'Light Vehicle pulling trailer': [2.3, 7.], # w: ~2.3m, l: ~7m
                        'Motor Coach bus': [2.55, 12.], # w: ~2.55m, l: ~12m
                        'Motorcycle or moped': [0.8, 2.0], # w: 0.64~1.0m, l: 1.8~2.5m
                        'Not applicable': [np.nan, np.nan],
                        'Object': [0., 0.],
                        'Other large construction equipment': [0., 0.],
                        'Other non-motorist': [0.5, 0.5],
                        'Other vehicle type': [1.75, 4.5],
                        'Pedestrian': [0.5, 0.5],
                        'Pickup truck': [2., 6.], # w: 1.9~2.1m, l: 5.4~6.4m
                        'Police': [1.9, 5.], # w: 1.8~2m, l: 4.8~5.1m
                        'School bus ': [2.5, 12.], # w: 2.4~2.6m, l: 10.7~13.7m
                        'Single-unit straight truck + trailer': [2.5, 15.], # w: 2.4~2.6m, l: 12~18m
                        'Single-unit straight truck, other': [2.4, 7.3], # w: 2.4~2.6m, l: 7.3~9.1m
                        'Single-unit straight truck: Box': [2.4, 7.3],
                        'Single-unit straight truck: Dump': [2.4, 7.3],
                        'Single-unit straight truck: Flatbed': [2.4, 7.3],
                        'Single-unit straight truck: Garbage/Recycling': [2.4, 7.3],
                        'Single-unit straight truck: Multistop/Step van': [2.4, 7.3],
                        'Single-unit straight truck: Tow truck': [2.6, 9.1],
                        'Sport Utility Vehicle': [1.85, 4.9], # w: 1.76~2m, l: 4.7~5.1m
                        'Tractor only': [1.95, 4.5], # w: 1.8~2.1m, l: 3.7~4.9m
                        'Tractor-trailer: Car carrier': [2.6, 19.], # w: ~2.6m, l: 16~21m
                        'Tractor-trailer: Dump trailer': [2.6, 12.], # w: ~2.6m, l: 10.7~13.7m
                        'Tractor-trailer: Enclosed box': [2.6, 15.], # w: ~2.6m, l: 14.6~16.2m
                        'Tractor-trailer: Flatbed': [2.6, 15.], # w: ~2.6m, l: 14.6~16.2m
                        'Tractor-trailer: Livestock': [2.6, 14.], # w: ~2.6m, l: 12.2~16.2m
                        'Tractor-trailer: Multiple box': [2.6, 16.2], # w: ~2.6m, l: ~16.2m
                        'Tractor-trailer: Tank': [2.6, 14.], # w: ~2.6m, l: 12.2~16.2m
                        'Transit bus': [2.55, 11.95], # w: ~2.55m, l: ~11.95m
                        'Unknown vehicle type': [1.75, 4.5],
                        'Van (minivan or standard van)': [1.9, 5.2]} # w: 1.9~2.1m, l: 4.8~5.9m
    vehicle_dimension = pd.DataFrame(vehicle_dimension, index=['width', 'length']).T
    return vehicle_dimension


# Combine metadata from both Honda Data Support and Driving Assistance Systems
print('Loading metadata...')
meta_das = pd.read_csv(path_raw + 'FileToUse/metadata_timeseries_video_DriverAssistanceSystems.csv') # 41,102 events
meta_honda = pd.read_csv(path_raw + 'FileToUse/metadata_timeseries_video_HondaDataSupport.csv') # 41,325 events

meta_both = meta_das[['event_id','time_series']].merge(meta_honda[['event_id','time_series']], on='event_id', how='outer', suffixes=('_das','_honda'))
meta_both['file_dir'] = path_raw_honda + 'Time Series Export/'
meta_both['file2use'] = meta_both['time_series_honda']
condition = meta_both['time_series_honda'].isnull()
meta_both.loc[condition,'file_dir'] = path_raw_das + 'Videos_And_TimeSeriesData/'
meta_both.loc[condition,'file2use'] = meta_both.loc[condition,'time_series_das']
meta_both = meta_both.set_index('event_id')  # 41,404 events


# Move the files to use to FileToUse folder
print('Moving files to FileToUse folder...')
os.makedirs(path_raw + 'FileToUse/TimeSeries/', exist_ok=True)
for event_id in tqdm(meta_both.index.values, desc='Moving files'):
    original_path = meta_both.loc[event_id,'file_dir'] + meta_both.loc[event_id,'file2use']
    if os.path.exists(original_path):
        new_path = path_raw + 'FileToUse/TimeSeries/' + meta_both.loc[event_id,'file2use']
        if not os.path.exists(new_path):
            shutil.copyfile(original_path, new_path)


# Categorise events
events = pd.read_csv(path_raw + 'FileToUse/InsightTables/Event_Table.csv')
severity = events[['eventID','eventSeverity1','eventSeverity2']].set_index('eventID')
categories = {'SafeBaseline': (severity['eventSeverity1'].isin(['Balanced-Sample Baseline', 'Additional Baseline'])), 
              'Crash': (severity['eventSeverity1']=='Crash')&(severity['eventSeverity2']=='Not Applicable'),
              'NearCrash': (severity['eventSeverity1']=='Near-Crash')&(severity['eventSeverity2']=='Not Applicable'), 
              'NearCrash-NearCrash': (severity['eventSeverity1']=='Near-Crash')&(severity['eventSeverity2']=='Near-Crash'),
              'SecondaryNearCrash': (severity['eventSeverity1']=='Non-Subject Conflict')&(severity['eventSeverity2']=='Near-Crash'),
              'NearCrash-CrashRelevant': (severity['eventSeverity1']=='Near-Crash')&(severity['eventSeverity2']=='Crash-Relevant'),
              'NearCrash-Crash': (severity['eventSeverity1']=='Near-Crash')&(severity['eventSeverity2']=='Crash'),
              'Crash-NearCrash': (severity['eventSeverity1']=='Crash')&(severity['eventSeverity2']=='Near-Crash'),
              'Crash-Crash': (severity['eventSeverity1']=='Crash')&(severity['eventSeverity2']=='Crash'),
              'CrashRelevant-NearCrash': (severity['eventSeverity1']=='Crash-Relevant')&(severity['eventSeverity2']=='Near-Crash'),
              'SecondaryCrash': (severity['eventSeverity1']=='Non-Subject Conflict')&(severity['eventSeverity2']=='Crash'),
              'NearCrash-OtherConflict': (severity['eventSeverity1']=='Near-Crash')&(severity['eventSeverity2']=='Non-Subject Conflict'),
              'Crash-OtherConflict': (severity['eventSeverity1']=='Crash')&(severity['eventSeverity2']=='Non-Subject Conflict'),
              'CrashRelevant-Crash': (severity['eventSeverity1']=='Crash-Relevant')&(severity['eventSeverity2']=='Crash'),
              'Crash-CrashRelevant': (severity['eventSeverity1']=='Crash')&(severity['eventSeverity2']=='Crash-Relevant'),
              }
for cat in categories:
    severity.loc[categories[cat], 'event_category'] = cat
meta_both['event_category'] = severity.loc[meta_both.index.values]['event_category'].values
print('Event category distribution:')
print(meta_both['event_category'].value_counts())


# Information about vehicles involved in events, we call the first involved as target, and the second involved as other
vehicles_involved = events[['eventID','eventNature1','eventNature2']].set_index('eventID').astype(str)
targets = {'(null)': 'none', 
           'Conflict with a following vehicle': 'following',
           'Conflict with a lead vehicle': 'leading',
           'Conflict with animal': 'animal',
           'Conflict with merging vehicle': 'merging',
           'Conflict with obstacle/object in roadway': 'obstacle',
           'Conflict with oncoming traffic': 'oncoming', 
           'Conflict with parked vehicle': 'parked',
           'Conflict with pedalcyclist': 'cyclist', 
           'Conflict with pedestrian': 'pedestrian',
           'Conflict with vehicle in adjacent lane': 'adjacent_lane',
           'Conflict with vehicle moving across another vehicle path (through intersection)': 'intersection_crossing',
           'Conflict with vehicle moving across another vehicle path (through intersection) ': 'intersection_crossing',
           'Conflict with vehicle turning across another vehicle path (opposite direction)': 'turning_across_opposite',
           'Conflict with vehicle turning across another vehicle path (same direction)': 'turning_across_parallel',
           'Conflict with vehicle turning into another vehicle path (opposite direction)': 'turning_into_opposite',
           'Conflict with vehicle turning into another vehicle path (same direction)': 'turning_into_parallel',
           'Other': 'unknown', 
           'Single vehicle conflict': 'single', 
           'Unknown conflict': 'unknown', 
           'nan': 'none'}
for target in targets:
    vehicles_involved.loc[vehicles_involved['eventNature1']==target, 'first'] = targets[target]
    vehicles_involved.loc[vehicles_involved['eventNature2']==target, 'second'] = targets[target]
meta_both[['first','second']] = vehicles_involved.loc[meta_both.index.values][['first','second']].values


# Vehicle dimensions, note that the `other` target may not be involved in a secondary event
vehicle_dimension = define_vehicle_dimension()
ego_vehicle = pd.read_csv(path_raw + 'FileToUse/InsightTables/VehicleDetailTable.csv')
ego_vehicle['width'] = ego_vehicle['classification'].map(vehicle_dimension['width'])
ego_vehicle['length'] = ego_vehicle['classification'].map(vehicle_dimension['length'])
meta_both[['ego_width','ego_length']] = ego_vehicle.set_index('eventID').loc[meta_both.index.values][['width','length']].values

events['target_width'] = events['motorist2Type'].map(vehicle_dimension['width'])
events['target_length'] = events['motorist2Type'].map(vehicle_dimension['length'])
events['other_width'] = events['motorist3Type'].map(vehicle_dimension['width'])
events['other_length'] = events['motorist3Type'].map(vehicle_dimension['length'])
meta_both[['target_width','target_length','other_width','other_length']] = events.set_index('eventID').loc[meta_both.index.values][['target_width','target_length','other_width','other_length']].values

condition = meta_both['event_category'].isin(['SecondaryNearCrash','SecondaryCrash'])
meta_both.loc[condition, ['target_width','target_length','other_width','other_length']] = meta_both.loc[condition, ['other_width','other_length','target_width','target_length']].values
'''
For secondary events, in general motorist3 is described as the target; but sometimes motorist2 is the target.
To try the best to keep width&length correct, we check specifically for animal, pedestrian, cyclist, obstacle, and single.
'''
non_veh_list = ['animal','pedestrian','cyclist','obstacle','single']
condition1 = (meta_both['first'].isin(non_veh_list))&(meta_both['target_width']>1.)
condition2 = (meta_both['second'].isin(non_veh_list))&(meta_both['other_width']>1.)
condition = condition&(condition1|condition2)
meta_both.loc[condition, ['target_width','target_length','other_width','other_length']] = meta_both.loc[condition, ['other_width','other_length','target_width','target_length']].values

os.makedirs(path_processed, exist_ok=True)
meta_both.to_csv(path_processed + 'metadata_birdseye.csv')

