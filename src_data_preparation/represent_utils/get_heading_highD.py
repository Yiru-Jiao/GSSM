'''
This script is reused from DataProcessing/utils/get_heading_ekf.py in UnifiedConflictDetection https://github.com/Yiru-Jiao/UnifiedConflictDetection
'''

import numpy as np
import pandas as pd


def ekf(params, data, track_id, return_loss=False):
    '''
    Perform extended Kalman filter for constant heading and velocity estimation.

    Parameters:
    - params: List of EKF parameters [uncertainty_init, uncertainty_pos, uncertainty_speed, noise_acc, noise_rad, noise_speed]
    - data: Pandas DataFrame containing vehicle trajectory data
    - track_id: Track ID of the vehicle
    - return_loss: Boolean flag indicating whether to return loss value

    Returns:
    - If return_loss is True, returns the loss value
    - Otherwise, returns the DataFrame with estimated vehicle states
    '''
    if len(params) == 0:  # default parameters if not provided
        uncertainty_init = 100.
        uncertainty_pos = 5.
        uncertainty_speed = 5.
        noise_acc = 4.
        noise_rad = 2.
        noise_speed = 50.
    else:
        uncertainty_init, uncertainty_pos, uncertainty_speed, noise_acc, noise_rad, noise_speed = params

    veh = data.loc[track_id].sort_values('frame_id').copy()

    # Downsample from 25 fps to 10 fps
    ## linear interpolation to make fps 50
    veh_new = pd.DataFrame(columns=veh.columns)
    new_frames = np.concatenate([veh['frame_id'].values, veh['frame_id'].values + 0.5])
    veh_new['frame_id'] = np.sort(new_frames)
    for var in ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'laneId', 'precedingId', 'followingId', 'direction']:
        veh_new[var] = np.interp(veh_new['frame_id'].values, veh['frame_id'].values, veh[var])
    ## downsample to fps 10
    veh_new['frame_id'] = (veh_new['frame_id'] * 2).astype(int)
    veh_new = veh_new[veh_new['frame_id'] % 5 == 0]
    veh_new['frame_id'] = (veh_new['frame_id'] / 5).astype(int).values
    veh_new['track_id'] = track_id
    veh_new['width'] = veh['width'].values[0]
    veh_new['length'] = veh['length'].values[0]
    veh_new['laneId'] = veh_new['laneId'].astype(int)
    veh_new['direction'] = veh_new['direction'].astype(int)
    for var, alternatives in zip(['precedingId', 'followingId'], [veh['precedingId'].unique(), veh['followingId'].unique()]):
        interped_ids = np.array([veh_new[var].values] * len(alternatives)).T
        original_ids = np.array([alternatives] * len(veh_new))
        veh_new[var] = alternatives[np.argmin(abs(interped_ids - original_ids), axis=1)]
    veh = veh_new.copy()

    # Initialize
    numstates = 4
    P = np.eye(numstates) * uncertainty_init  # Initial Uncertainty
    dt = np.diff(veh['frame_id'].values) / 10
    dt = np.hstack([dt[0], dt])
    R = np.diag([uncertainty_pos, uncertainty_pos, uncertainty_speed])  # Measurement Noise
    I = np.eye(numstates)
    mx, my, mv = veh['x'].values, veh['y'].values, np.sqrt(veh['vx'] ** 2 + veh['vy'] ** 2).values
    dx = np.hstack([0, np.diff(veh['x'])])
    dy = np.hstack([0, np.diff(veh['y'])])
    ds = np.sqrt(dx ** 2 + dy ** 2)
    Trigger = (ds != 0.0).astype('bool')  # Perform EKF when there is a change in position

    # Measurement vector
    measurements = np.vstack((mx, my, mv))
    m = measurements.shape[1]

    head = veh[['x', 'y']].diff(2)
    head = head[~np.isnan(head['x'])]
    head = head[(head['x'] != 0) | (head['y'] != 0)].values
    if len(head) == 0:
        estimates = np.zeros((m, 4)) * np.nan
    else:
        psi0 = np.arctan2(head[0][1], head[0][0])
        x = np.array([mx[0], my[0], mv[0], psi0])  # Initial State

        # Estimated vector
        estimates = np.zeros((m, 4))

        for filterstep in range(m):
            # Time Update (Prediction)
            x[0] = x[0] + dt[filterstep] * x[2] * np.cos(x[3])
            x[1] = x[1] + dt[filterstep] * x[2] * np.sin(x[3])
            x[2] = x[2]
            x[3] = (x[3] + np.pi) % (2.0 * np.pi) - np.pi

            # Calculate the Jacobian of the Dynamic Matrix A
            a13 = dt[filterstep] * np.cos(x[3])
            a14 = -dt[filterstep] * x[2] * np.sin(x[3])
            a23 = dt[filterstep] * np.sin(x[3])
            a24 = dt[filterstep] * x[2] * np.cos(x[3])
            JA = np.matrix([[1.0, 0.0, a13, a14],
                            [0.0, 1.0, a23, a24],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], dtype=float)

            # Calculate the Process Noise Covariance Matrix
            s_pos = 0.5 * noise_acc * dt[filterstep] ** 2
            s_psi = noise_rad * dt[filterstep]
            s_speed = noise_speed * dt[filterstep]

            Q = np.diag([s_pos ** 2, s_pos ** 2, s_speed ** 2, s_psi ** 2])

            # Project the error covariance ahead
            P = JA * P * JA.T + Q

            # Measurement Update (Correction)
            hx = np.matrix([[x[0]], [x[1]], [x[2]]])

            if Trigger[filterstep]:
                JH = np.matrix([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0]], dtype=float)
            else:
                JH = np.matrix([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]], dtype=float)

            S = JH * P * JH.T + R
            K = (P * JH.T) * np.linalg.inv(S.astype('float'))

            # Update the estimate via
            Z = measurements[:, filterstep].reshape(JH.shape[0], 1)
            y = Z - (hx)  # Innovation or Residual
            x = x + np.array(K * y).reshape(-1)

            # Update the error covariance
            P = (I - (K * JH)) * P

            # Save states
            estimates[filterstep, :] = x

    veh[['x_kf', 'y_kf', 'speed_kf', 'psi_kf']] = estimates

    if return_loss:
        veh4loss = veh[veh['frame_id'] % 2 == 0]  # these are the frames that have original data
        pos_rmse = np.sqrt(((veh4loss['x'] - veh4loss['x_kf']) ** 2 + (veh4loss['y'] - veh4loss['y_kf']) ** 2).mean())
        speed_rmse = np.sqrt((((np.sqrt(veh4loss['vx'] ** 2 + veh4loss['vy'] ** 2)) - veh4loss['speed_kf']) ** 2).mean())

        return 0.2 * pos_rmse + 0.8 * speed_rmse
    else:
        return veh