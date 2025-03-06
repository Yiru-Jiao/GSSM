'''
This script preprocesses the Argoverse2 dataset. 
The Argoverse2 dataset consists of scenario data stored in zarr format.
'''

import os
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

path_raw = './RawData/'
path_processed = './ProcessedData/Argoverse/'
os.makedirs(path_processed, exist_ok=True)


def get_df(motion, time_seq, veh_type):
    df = pd.DataFrame(motion, columns=['x','y','vx','vy','ax','ay','psi'])
    df['time'] = time_seq
    df['v'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['hx'] = np.cos(df['psi'])
    df['hy'] = np.sin(df['psi'])
    df['acc'] = df['ax']*df['hx'] + df['ay']*df['hy']
    df['veh_type'] = veh_type
    return df[['time','x','y','v','vx','vy','hx','hy','acc','veh_type']]


def read_scenario(filename, root, initial_target_id, return_map=False):
    # There are multiple objects in each scenario, we are interested in the first two objects which are interacting
    dt = zarr.open(root+filename, mode='r')
    slices = dt['index'][:]
    timestep = dt['timestep'][:]
    motion = dt['motion'][:]
    veh_type = dt['category'][:]

    # In the cases involving an AV, the AV is always the first object in the scenario
    t_ego = timestep[slices[0]:slices[1]]
    motion_ego = motion[slices[0]:slices[1]]
    df_ego = get_df(motion_ego, t_ego, veh_type[0])
    data = []
    for i in range(1, len(veh_type)):
        t_sur = timestep[slices[i]:slices[i+1]]
        motion_sur = motion[slices[i]:slices[i+1]]
        df_sur = get_df(motion_sur, t_sur, veh_type[i])
        df_sur['target_id'] = i
        df = df_ego.merge(df_sur, on='time', suffixes=('_ego','_sur'), how='inner')
        data.append(df)
    data = pd.concat(data, ignore_index=True)
    data['target_id'] = data['target_id'] + initial_target_id

    if return_map:
        maps = dt.lane[:]
        return data, maps
    else:
        return data


# Set the root path for raw data
# if you downloaded the data from 4TU, firstly unzip the data
for ego_vtype in ['av','hv']:
    root = path_raw + f'Argoverse2/data_3m/{ego_vtype}/'
    if os.path.exists(path_processed + f'argo_{ego_vtype}.h5'):
        print(f'Processing {ego_vtype} already done. Skipping...')
        continue

    print(f'Processing {ego_vtype}...')
    # Get the list of files from the root directory
    filenames = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    data = []
    initial_target_id = -1
    for idx in tqdm(range(len(filenames))):
        filename = filenames[idx]
        df = read_scenario(filename, root, initial_target_id)
        df['log_id'] = idx
        initial_target_id = df['target_id'].max()
        data.append(df)

    data = pd.concat(data, ignore_index=True)
    data[['log_id', 'target_id']] = data[['log_id', 'target_id']].astype(int)
    # Set vehicle dimensions
    data['length_ego'] = 4.5
    data['width_ego'] = 1.75
    veh_types = [0, 1, 2, 3, 4, 10] # 0: HV, 1: Pedestrian, 2: Motorcyclist, 3: Cyclist, 4: Bus, 10: Automated Vehicle
    widths = [1.75, 0.5, 0.8, 0.5, 2.55, 1.75]
    lengths = [4.5, 0.5, 2.0, 1.5, 11.95, 4.5]
    for veh_type, width, length in zip(veh_types, widths, lengths):
        data.loc[data['veh_type_sur']==veh_type, 'length_sur'] = length
        data.loc[data['veh_type_sur']==veh_type, 'width_sur'] = width
    data.loc[data['length_sur'].isna(), 'length_sur'] = 4.5
    data.loc[data['width_sur'].isna(), 'width_sur'] = 1.75
    data = data.drop(columns=['veh_type_ego','veh_type_sur'])

    # Saving the computed data to an HDF5 file
    data.to_hdf(path_processed + f'argo_{ego_vtype}.h5', key='data', mode='w')
