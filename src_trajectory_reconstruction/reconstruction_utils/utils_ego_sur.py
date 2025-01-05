'''
This script contains functions for data processing.
'''
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
from reconstruction_utils.utils_ekf import reconstruct_ego, reconstruct_surrounding


# Create dataframes for ego vehicle and surrounding vehicles
def create_dataframe(sample, event_id, target_id=0):
    df_ego = pd.DataFrame({'event_id': event_id,
                           'timestamp': sample['vtti.timestamp'].values.astype(int), # unit: ms
                           'time': np.round(sample['vtti.timestamp'].values/1000,1), # unit: s
                           'speed_comp': sample['vtti.speed_network'].values/3.6, # unit: m/s
                           'yaw_rate': -sample['vtti.gyro_z'].values, # unit: deg/s, positive for right turn in original data, for left turn after adjustment
                           'acc_lat': sample['vtti.accel_y'].values, # unit: g, positive for right turn
                           'acc_lon': sample['vtti.accel_x'].values, # unit: g, positive for forward
                           'brake': sample['vtti.pedal_brake_state'].values, # 0=off, 1=on, 2/3=unknown
                           'wheel_steering': sample['vtti.steering_wheel_position'].values, # unit: deg, positive for ? 
                           'turn_signal': sample['vtti.turn_signal'].values, # 0=off, 1=left, 2=right, 3=both, 254/255=unknown
                           })
    df_ego = df_ego.drop_duplicates(subset=['timestamp'], keep='first')
    ## remove rows at the beginning with missing values
    df_ego = df_ego.iloc[np.where(np.all(~df_ego[['speed_comp','acc_lat','acc_lon']].isna(), axis=1))[0][0]:]
    df_ego['event_id'] = event_id
    df_ego[['event_id','timestamp']] = df_ego[['event_id','timestamp']].astype(int)

    df_forward = [pd.DataFrame()]
    track_ids = ['TRACK'+str(i)+'_TARGET_ID' for i in range(1,9)]
    targets = np.unique(sample[track_ids])
    for target in targets[targets>0]:
        id_mask = np.where(sample[track_ids].values==target)
        timestamp = sample['vtti.timestamp'].values[id_mask[0]] # unit: ms
        time = np.round(timestamp/1000,1)
        local_dy = sample[['TRACK'+str(i)+'_X_POS_PROCESSED' for i in range(1,9)]].values[id_mask] # longitudinal, unit: m
        local_dx = sample[['TRACK'+str(i)+'_Y_POS_PROCESSED' for i in range(1,9)]].values[id_mask] # lateral, unit: m
        delta_vy = sample[['TRACK'+str(i)+'_X_VEL_PROCESSED' for i in range(1,9)]].values[id_mask] # longitudinal relative velocity, unit: m/s
        delta_vx = sample[['TRACK'+str(i)+'_Y_VEL_PROCESSED' for i in range(1,9)]].values[id_mask] # lateral relative velocity, unit: m/s
        df_target = pd.DataFrame({'event_id': event_id,
                                  'target_id': target_id,
                                  'timestamp': timestamp.astype(int),
                                  'time': time,
                                  'local_dx': -local_dx, # positive for left in the original data, adjusted for right
                                  'local_dy': local_dy,
                                  'delta_vx': -delta_vx, # positive for left in the original data, adjusted for right
                                  'delta_vy': delta_vy})
        df_target = df_target.drop_duplicates(subset=['timestamp'], keep='first')
        df_target['event_id'] = event_id
        df_target['target_id'] = target_id
        df_target[['event_id','target_id','timestamp']] = df_target[['event_id','target_id','timestamp']].astype(int)
        df_forward.append(df_target)
        target_id += 1
    df_forward = pd.concat(df_forward)

    reconnected_ids = dict()
    if len(df_forward)>0:
        targets = df_forward.groupby('target_id')['time'].count()
        targets = targets[targets>=2].index.sort_values()
        if len(targets)>1:
            df_forward = df_forward.sort_values(['target_id','time']).set_index('target_id')
            for target in targets[1:]:
                if target not in df_forward.index:
                    continue
                current_sur = df_forward.loc[target].iloc[0]
                previous_sur = df_forward[(df_forward['time']>=current_sur['time']-0.2)&
                                          (df_forward['time']<=current_sur['time']+0.2)&
                                          (df_forward.index<target)]
                if len(previous_sur)>0:
                    pos_diff = np.sqrt((previous_sur['local_dx']-current_sur['local_dx'])**2 +
                                       (previous_sur['local_dy']-current_sur['local_dy'])**2)
                    dist_threshold = min(0.5, max(2.5, np.sqrt((current_sur['delta_vy']**2+current_sur['delta_vx']**2))*0.3))
                    if pos_diff.min()<=dist_threshold:
                        reconnected_ids[target] = pos_diff.idxmin()
        if len(reconnected_ids)>0:
            while df_forward.index.isin(reconnected_ids.keys()).any():
                df_forward = df_forward.rename(index=reconnected_ids)
            df_forward = df_forward.reset_index()
            df_forward = df_forward.drop_duplicates(subset=['event_id','target_id','timestamp'], keep='first')
        else:
            df_forward = df_forward.reset_index()

    return df_ego, df_forward, target_id, reconnected_ids



# reconstruct trajectory of the ego vehicle
def process_ego(df_ego, event_id, pdf=None, ego_params=None):
    ## constants
    g = 9.81  ### gravity, m/s^2
    
    ## convert yaw rate to radians per second
    df_ego['yaw_rate'] = np.deg2rad(df_ego['yaw_rate'])
    ## convert acceleration to m/s^2
    df_ego['acc_lat'] = df_ego['acc_lat'] * g
    df_ego['acc_lon'] = df_ego['acc_lon'] * g

    if ego_params is None:
        ego_params = {'uncertainty_init':20.,
                      'uncertainty_speed':8.,
                      'uncertainty_omega':0.05,
                      'uncertainty_acc':4.,
                      'max_jerk':15.,
                      'max_yaw_rate':0.5,
                      'max_acc':9.8,
                      'max_yaw_acc':np.pi}
    
    df_ego_original = df_ego.copy()
    ## make time interval equal
    time_frequencies = pd.DataFrame(np.round(np.arange(df_ego['time'].min(), df_ego['time'].max(), 0.1),1), columns=['time'])
    df_ego = time_frequencies.merge(df_ego, on='time', how='outer').sort_values('time')
    df_ego.loc[df_ego['timestamp'].isna(), 'timestamp'] = -1
    df_ego.loc[df_ego['event_id'].isna(), 'event_id'] = event_id
    ## interpolate missing values
    for var in ['speed_comp','yaw_rate','acc_lat','acc_lon']:
        if np.any(df_ego[var].isna()):
            valid = np.logical_not(df_ego[var].isna())
            interpolated = np.interp(df_ego['time'], df_ego['time'][valid], df_ego[var][valid])
            df_ego[var] = interpolated
    df_ego = df_ego[df_ego['time'].isin(time_frequencies['time'])]

    valid_start = np.all(df_ego['speed_comp'].iloc[:5]>=0)
    valid_end = np.all(df_ego['speed_comp'].iloc[-5:]>=0)
    if valid_start and not valid_end:
        reverse = False
        df_ego = reconstruct_ego(df_ego, ego_params, reverse=False)
    elif valid_end and not valid_start:
        reverse = True
        df_ego = reconstruct_ego(df_ego, ego_params, reverse=True)
    elif not valid_start and not valid_end:
        reverse = False
    elif valid_start and valid_end:
        df_order = reconstruct_ego(df_ego, ego_params, reverse=False)
        df_reverse = reconstruct_ego(df_ego, ego_params, reverse=True)
        to_count = (df_ego['speed_comp']>=0).values
        error_order = np.sum(np.abs(df_order['v_ekf'] - df_order['speed_comp']).values[to_count])
        error_order += np.sum(np.abs(df_order['omega_ekf'] - df_order['yaw_rate']).values[to_count])
        error_reverse = np.sum(np.abs(df_reverse['v_ekf'] - df_reverse['speed_comp']).values[to_count])
        error_reverse += np.sum(np.abs(df_reverse['omega_ekf'] - df_reverse['yaw_rate']).values[to_count])
        if error_order < error_reverse + to_count.sum()*0.02:
            reverse = False
            df_ego = df_order.copy()
            df_order = None
        else:
            reverse = True
            df_ego = df_reverse.copy()
            df_reverse = None
    
    ## plot and save reconstructed trajectory
    if pdf is None:
        return df_ego
    else:
        fig, axes = plt.subplots(1, 3, figsize=(8, 1.8))
        if valid_start or valid_end:
            axes[0].plot(df_ego['time'], df_ego['v_ekf'], marker='o', markersize=3, color='tab:blue', rasterized=True)
            axes[1].plot(df_ego['time'], df_ego['psi_ekf'], marker='o', markersize=3, color='tab:blue', rasterized=True)
            axes[2].plot(df_ego['time'], df_ego['acc_ekf'], marker='o', markersize=3, label='ekf', color='tab:blue', rasterized=True)
        axes[0].plot(df_ego_original['time'], df_ego_original['speed_comp'], alpha=0.5, marker='o', markersize=1, color='tab:orange', rasterized=True)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_title('Speed (m/s)')
        yaw = (np.cumsum(df_ego_original['yaw_rate']*np.gradient(df_ego_original['time']))).values
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        if reverse:
            yaw = yaw-yaw[-1]
        axes[1].set_ylim(-np.pi, np.pi)
        axes[1].plot(df_ego_original['time'], yaw, alpha=0.5, marker='o', markersize=1, color='tab:orange', rasterized=True)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Yaw (rad)')
        axes[2].plot(df_ego_original['time'], df_ego_original['acc_lon'], alpha=0.5, marker='o', markersize=1, label='raw', color='tab:orange', rasterized=True)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Acceleration (m/s^2)')
        axes[2].legend(loc='lower left')

        if valid_start and valid_end:
            fig.suptitle('Event id: '+str(event_id)+', Reverse: '+str(reverse)+', Error in order: '+str(round(error_order,2))+', Error in reverse: '+str(round(error_reverse,2)),
                        y=1.08)
        else:
            fig.suptitle('Event id: '+str(event_id)+', Reverse: '+str(reverse), y=1.08)

        pdf.attach_note('Event id: '+str(event_id))
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        fig.clear()
        plt.close(fig)  # Close the figure after saving it to free up memory
        
        return df_ego, valid_start|valid_end, pdf



# Rotate (x2t, y2t) to the coordinate system with the y-axis along (xyaxis, yyaxis)
def rotate_coor(xyaxis, yyaxis, x2t, y2t):
    x = yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t-xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
    y = xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t+yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
    return x, y


# Apply Extended Kalman Filter to a surrounding vehicle
def ekf_sur(target_id, df_sur, sur_params):
    df_target = df_sur.loc[target_id].copy()

    if len(df_target.values.shape)<2: # the target has only one row
        return pd.DataFrame()
    elif len(df_target.values) < 10: # ignore vehicles appearing less than 1 second
        return pd.DataFrame()
    else:
        df_target = df_target.reset_index()
        event_id = df_target['event_id'].values[0]
        ## make time interval equal
        time_frequencies = pd.DataFrame(np.round(np.arange(df_target['time'].min(), df_target['time'].max(), 0.1),1), columns=['time'])
        df_target = time_frequencies.merge(df_target, on='time', how='outer').sort_values('time')
        na_rows = df_target['timestamp'].isna()
        df_target.loc[na_rows, ['timestamp','event_id','target_id']] = np.repeat([-1,event_id,target_id], na_rows.sum()).reshape(3,-1).T
        ## interpolate missing values
        for var in ['x','y','speed_comp']:
            if np.any(df_target[var].isna()):
                valid = np.logical_not(df_target[var].isna())
                interpolated = np.interp(df_target['time'], df_target['time'][valid], df_target[var][valid])
                df_target[var] = interpolated
        df_target = df_target[df_target['time'].isin(time_frequencies['time'])]

        df_target = reconstruct_surrounding(df_target, sur_params)
        return df_target


# Process surrounding vehicles
def process_surrounding(df_ego, df_sur, ego_length, sur_params=None, n_jobs=4):
    df_sur = df_sur[df_sur['time'].isin(df_ego['time'])].copy()
    df_ego_sur = df_ego.set_index('time').loc[df_sur['time'].values].reset_index()
    heading_ego = np.array([np.cos(df_ego_sur['psi_ekf'].values), np.sin(df_ego_sur['psi_ekf'].values)]).T

    ## forward only as SHRP2 data does not provide rearward information
    point_head = df_ego_sur[['x_ekf','y_ekf']].values + heading_ego*ego_length/2
    ego_reference_x = point_head[:,0]
    ego_reference_y = point_head[:,1]

    ref_x, ref_y = rotate_coor(np.cos(df_ego_sur['psi_ekf'].values),
                            np.sin(df_ego_sur['psi_ekf'].values), 0, 1)
    global_dx, global_dy = rotate_coor(ref_x, ref_y, df_sur['local_dx'].values, df_sur['local_dy'].values)
    df_sur['x'] = global_dx + ego_reference_x
    df_sur['y'] = global_dy + ego_reference_y
    delta_vx, delta_vy = df_sur['delta_vx'], df_sur['delta_vy']
    df_sur['speed_comp'] = np.sqrt(delta_vx**2 + (df_ego_sur['v_ekf'].values + delta_vy)**2)
    
    if sur_params is None:
        sur_params = {'uncertainty_init':15.,
                      'uncertainty_pos':2.,
                      'uncertainty_speed':8.,
                      'max_acc':9.8,
                      'max_yaw_rate':0.5}

    df_sur = df_sur.sort_values(['target_id','time']).set_index('target_id')
    df_sur_ekf = Parallel(n_jobs=n_jobs)(delayed(ekf_sur)(target_id, df_sur, sur_params) for target_id in df_sur.index.unique())
    if len(df_sur_ekf)>0:
        df_sur_ekf = pd.concat(df_sur_ekf)
    else:
        df_sur_ekf = pd.DataFrame()
        
    return df_sur_ekf
    
