'''
This script preprocesses the highD dataset.

This reuses two scripts from UnifiedConflictDetection https://github.com/Yiru-Jiao/UnifiedConflictDetection
- DataProcessing/preprocessing_highD.py
- DataProcessing/extraction_highD_LC.py

Agent type (car/truck/cyclist/pedestrian) is implicitly defined by width and length. 
The width of a car/truck is at least 1.5m, and the length is at least 3.5m.
'''

import os
import sys
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import represent_utils.get_heading_highD as ekf

path_raw = './RawData/'
path_processed = './ProcessedData/highD/'
os.makedirs(path_processed, exist_ok=True)


class LaneChangeExtractor():
    '''
    Extract lane changes from the highD dataset.
    '''
    def __init__(self, data, initial_lc_id=0):
        super().__init__()
        data['time'] = data['frame_id']/10
        data = data.drop(columns=['frame_id'])
        data = data.sort_values(['track_id','time']).set_index('track_id')
        self.data = data
        lane_change = data.groupby('track_id')['laneId'].nunique() > 1
        self.lc_track_ids = lane_change.index[lane_change].values
        self.initial_lc_id = initial_lc_id

    def extract_lanechange(self,):
        lc_id = self.initial_lc_id
        lane_changes = []
        for track_id in tqdm(self.lc_track_ids):
            veh = self.data.loc[track_id]
            preceding_vehs = veh['precedingId'][veh['precedingId']>0].unique()
            following_vehs = veh['followingId'][veh['followingId']>0].unique()
            if len(preceding_vehs)==0 or len(following_vehs)==0:
                continue # skip if the vehicle changed lane without interacting with other vehicles
            for interact_veh_id in np.concatenate([preceding_vehs, following_vehs]):
                veh_ego = self.data.loc[track_id].drop(columns=['laneId','precedingId', 'followingId'])
                veh_sur = self.data.loc[interact_veh_id].drop(columns=['laneId','precedingId', 'followingId'])
                df = veh_ego.merge(veh_sur, on='time', suffixes=('_ego', '_sur'), how='inner')
                if len(df)<25:
                    continue # skip if the interaction is shorter than 2.5 seconds
                df['lc_id'] = lc_id
                df['track_id_ego'] = track_id
                df['track_id_sur'] = interact_veh_id
                lane_changes.append(df)
                lc_id += 1
        return pd.concat(lane_changes, ignore_index=True)


if __name__ == '__main__':
    # Extract meta information
    metadatafiles =  sorted(glob.glob(path_raw + 'highD/RecordingMetadata/*.csv'))
    metadata = []
    for metadatafile in metadatafiles:
        df = pd.read_csv(metadatafile)
        metadata.append(df)
    metadata = pd.concat(metadata)
    metadata['lane_num'] = metadata.lowerLaneMarkings.str.len()//5
    metadata['numFrames'] = (metadata['frameRate']*metadata['duration']).astype(int)

    print(metadata.groupby('locationId').agg({'numCars':'sum','numTrucks':'sum'}))
    trackid_base = 10**len(str(int(metadata['numCars'].max())))
    frameid_base = 10**len(str(int(metadata['numFrames'].max())))


    ekf_params = np.array([100.2683, 0.01, 0.01, 11.1333, 2.5, 52.4380])
    print('Processing order:', metadata.locationId.unique())
    for locid in tqdm(metadata.locationId.unique(), desc='location'):
        loc = 'highD_' + str(locid).zfill(2)
        data_files = [str(id).zfill(2) + '_tracks' for id in metadata[(metadata.locationId==locid)].id.values]
        metadata_files = [str(id).zfill(2) + '_tracksMeta' for id in metadata[(metadata.locationId==locid)].id.values]
        data = []

        for data_file, metadata_file in tqdm(zip(data_files, metadata_files), total=len(data_files), desc='file'):
            file_id = int(data_file[:2])
            df = pd.read_csv(path_raw + 'highD/' + data_file +'.csv')
            meta = pd.read_csv(path_raw + 'highD/' + metadata_file +'.csv')
            df = df.rename(columns={'frame':'frame_id',
                                    'id':'track_id',
                                    'xVelocity':'vx',
                                    'yVelocity':'vy',
                                    'xAcceleration':'ax',
                                    'yAcceleration':'ay',
                                    'width':'length',
                                    'height':'width'})
            df['direction'] = meta.set_index('id').reindex(df['track_id'].values)['drivingDirection'].values
            df = df[['track_id','frame_id','x','y','vx','vy','ax','ay','width','length',
                    'laneId','precedingId','followingId','direction']]
                    
            # move the coordinates to the center of the vehicles, we don't consider the angle because
            # 1) the bounding box is not sure to have been detected along the heading direction
            # 2) on highway event lane-changes do not have large angles between the lanes
            # 3) further processing will be conducted later to restimate the heading and position
            df['x'] = df['x'] + df['length']/2
            df['y'] = df['y'] + df['width']/2
            # downsample from 25 fps to 10 fps and obtain heading direction using extended kalman filter
            track_ids = df['track_id'].unique()
            df = df.set_index('track_id')
            df = pd.concat(Parallel(n_jobs=4)(delayed(ekf.ekf)(ekf_params, df, track_id, False) for track_id in track_ids)).reset_index(drop=True)
            df['vx'] = df['speed_kf']*np.cos(df['psi_kf'])
            df['vy'] = df['speed_kf']*np.sin(df['psi_kf'])
            df['hx'] = np.cos(df['psi_kf'])
            df['hy'] = np.sin(df['psi_kf'])
            df = df[['track_id','frame_id','x_kf','y_kf', 'psi_kf', 'speed_kf',
                    'vx','vy','ax','ay','hx','hy','width','length',
                    'laneId','precedingId','followingId','direction']].rename(columns={'x_kf':'x','y_kf':'y','psi_kf':'psi_rad','speed_kf':'speed'})
            # redefine indcies to be unique for later data combination
            for var in ['track_id','precedingId','followingId']:
                df.loc[df[var]>0.5,var] = file_id*trackid_base+df.loc[df[var]>0.5,var].values
                df[var] = df[var].astype(int)
            df['frame_id'] = (file_id*frameid_base+df['frame_id']).astype(int)
            data.append(df)

        pd.concat(data).reset_index(drop=True).to_hdf(path_processed + loc + '.h5', key='data')
        data = []


    # Extract lane changes
    os.makedirs(path_processed+'lane_changing/', exist_ok=True)
    initial_lc_id = 0
    for loc_id in range(1,7):
        print('Extracting lane changes at location ' + str(loc_id) + '...')
        data = pd.read_hdf(path_processed+'highD_0'+str(loc_id)+'.h5', key='data')
        data['acc'] = data['ax']*data['hx'] + data['ay']*data['hy']
        data = data.rename(columns={'speed':'v'})
        lce = LaneChangeExtractor(data, initial_lc_id)
        lane_change = lce.extract_lanechange()
        lane_change = lane_change.drop_duplicates(subset=['track_id_ego','track_id_sur','time'])
        initial_lc_id = lane_change['lc_id'].max() + 1
        # Mirror the coordinates as in highD the y-axis points downwards
        lane_change = lane_change.rename(columns={'x_ego':'y_ego', 'y_ego':'x_ego', 'x_sur':'y_sur', 'y_sur':'x_sur',
                                                'vx_ego':'vy_ego', 'vy_ego':'vx_ego', 'vx_sur':'vy_sur', 'vy_sur':'vx_sur',
                                                'ax_ego':'ay_ego', 'ay_ego':'ax_ego', 'ax_sur':'ay_sur', 'ay_sur':'ax_sur',
                                                'hx_ego':'hy_ego', 'hy_ego':'hx_ego', 'hx_sur':'hy_sur', 'hy_sur':'hx_sur'})
        lane_change.to_hdf(path_processed+'lane_changing/lc_0'+str(loc_id)+'.h5', key='data')

    sys.exit(0)
