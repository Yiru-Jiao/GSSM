'''
This script preprocesses the Argoverse2 dataset. 
The Argoverse2 dataset consists of scenario data stored in zarr format.
'''

import os
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

path_raw = './RawData/Argoverse2/'
path_processed = './ProcessedData/Argoverse/'
os.makedirs(path_processed, exist_ok=True)


def get_df(motion, time_seq, veh_type):
    df = pd.DataFrame(motion, columns=['x','y','vx','vy','ax','ay','psi'])
    df['time'] = time_seq
    df['v'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['acc'] = np.sqrt(df['ax']**2 + df['ay']**2)
    df['hx'] = np.cos(df['psi'])
    df['hy'] = np.sin(df['psi'])
    df['veh_type'] = veh_type
    return df[['time','x','y','v','vx','vy','acc','hx','hy','veh_type']]


def read_scenario(filename, root, return_map=False):
    # There are multiple objects in each scenario, we are interested in the first two objects which are interacting
    dt = zarr.open(root+filename, mode='r')
    slices = dt['index'][:]
    timestep = dt['timestep'][:]
    motion = dt['motion'][:]
    veh_type = dt['category'][:]

    t_ego = timestep[slices[0]:slices[1]]
    motion_ego = motion[slices[0]:slices[1]]
    df_ego = get_df(motion_ego, t_ego, veh_type[0])
    t_sur = timestep[slices[1]:slices[2]]
    motion_sur = motion[slices[1]:slices[2]]
    df_sur = get_df(motion_sur, t_sur, veh_type[1])
    df = df_ego.merge(df_sur, on='time', suffixes=('_ego','_sur'), how='inner')

    if return_map:
        maps = dt.lane[:]
        return df, maps
    else:
        return df


# Set the root path for raw data
# if you downloaded the data from 4TU, firstly unzip the data
root = path_raw + 'data_3m/hv/'

print('Processing ...')
# Get the list of files from the root directory
filenames = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
data = []
for idx in tqdm(range(len(filenames))):
    filename = filenames[idx]
    df = read_scenario(filename, root)
    df['log_id'] = idx
    data.append(df)

data = pd.concat(data, ignore_index=True)
# Set vehicle dimensions
data['length_ego'] = 4.5
data['width_ego'] = 1.75
veh_types = [0, 1, 2, 3, 4] # 0: HV, 1: Pedestrian, 2: Motorcyclist, 3: Cyclist, 4: Bus
widths = [1.75, 0.5, 0.8, 0.5, 2.55]
lengths = [4.5, 0.5, 2.0, 1.5, 11.95]
for veh_type, width, length in zip(veh_types, widths, lengths):
    data.loc[data['veh_type_sur']==veh_type, 'length_sur'] = length
    data.loc[data['veh_type_sur']==veh_type, 'width_sur'] = width
data.loc[data['length_sur'].isna(), 'length_sur'] = 4.5
data.loc[data['width_sur'].isna(), 'width_sur'] = 1.75
data = data.drop(columns=['veh_type_ego','veh_type_sur'])

# Saving the computed data to an HDF5 file
print(data.head())
data.to_hdf(path_processed + 'argo_hv.h5', key='data', mode='w')
