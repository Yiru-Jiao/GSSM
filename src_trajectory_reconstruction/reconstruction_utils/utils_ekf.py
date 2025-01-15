'''
This script contains functions to reconstruct the ego vehicle and
surrounding vehicles using Extended Kalman Filter (EKF).
'''

import numpy as np


# Reconstruct the trajectory of the ego/subject vehicle
# Extended Kalman Filter for Constant Turn Rate and Acceleration,
# adapted from https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRA.ipynb
def reconstruct_ego(df_ego, params, reverse=False):
    uncertainty_init = params['uncertainty_init']
    uncertainty_speed = params['uncertainty_speed']
    uncertainty_omega = params['uncertainty_omega']
    uncertainty_acc = params['uncertainty_acc']
    max_jerk = params['max_jerk']
    max_yaw_rate = params['max_yaw_rate']
    max_acc = params['max_acc']
    max_yaw_acc = params['max_yaw_acc']

    veh = df_ego.sort_values('time').copy().reset_index(drop=True)
    if reverse:
        veh = veh.iloc[::-1].reset_index(drop=True)

    ## Initialize
    numstates = 6
    P = np.eye(numstates)*uncertainty_init # Initial Uncertainty
    R = np.diag([uncertainty_speed,uncertainty_omega,uncertainty_acc]) # Measurement Noise
    I = np.eye(numstates)
    dt = np.gradient(veh['time'])

    ## Measurement vector
    mv = veh['speed_comp'].values
    momega = veh['yaw_rate'].values
    momega[(momega<1e-4)&(momega>=0)] = 1e-4 ## Driving straight
    momega[(momega>-1e-4)&(momega<0)] = -1e-4
    macc = veh['acc_lon'].values
    ## Correct invalid measurements
    ### 1) the speed measurement is negative
    mv[mv<0.] = 0.
    ### 2) positive acceleration while the speed measurement drops to 0
    macc[(macc>0.)&(mv<=0.)] = 0.
    measurements = np.vstack((mv,momega,macc))
    m = measurements.shape[1] 

    ## Initial state
    x = np.array([0,0,0,mv[0],momega[0],macc[0]])

    ## Estimated vector
    estimates = np.zeros((m,numstates))
    estimates[0,:] = x

    for filterstep in np.arange(1,m):
        ## Time Update (Prediction)
        x[0] = x[0] + (1/x[4]**2) * (-x[3]*x[4]*np.sin(x[2]) -x[5]*np.cos(x[2]) +
                                     x[5]*np.cos(x[2]+x[4]*dt[filterstep]) + 
                                     (x[5]*x[4]*dt[filterstep]+x[3]*x[4])*np.sin(x[2]+x[4]*dt[filterstep]))
        x[1] = x[1] + (1/x[4]**2) * (x[3]*x[4]*np.cos(x[2]) - x[5]*np.sin(x[2]) +
                                     x[5]*np.sin(x[2]+x[4]*dt[filterstep]) +
                                     (-x[5]*x[4]*dt[filterstep]-x[3]*x[4])*np.cos(x[2]+x[4]*dt[filterstep]))
        x[2] = (x[2] + x[4] * dt[filterstep] + np.pi) % (2.0 * np.pi) - np.pi
        x[3] = x[3] + x[5] * dt[filterstep]
        x[4] = x[4]
        x[5] = x[5]

        ## Calculate the Jacobian of the Dynamic Matrix JA
        a13 = (-x[4]*x[3]*np.cos(x[2]) + x[5]*np.sin(x[2]) - x[5]*np.sin(dt[filterstep]*x[4]+x[2]) +
               (dt[filterstep]*x[4]*x[5]+x[4]*x[3])*np.cos(dt[filterstep]*x[4]+x[2])) / x[4]**2
        a14 = (-x[4]*np.sin(x[2]) + x[4]*np.sin(dt[filterstep]*x[4]+x[2])) / x[4]**2
        a15 = (-dt[filterstep]*x[5]*np.sin(dt[filterstep]*x[4]+x[2]) + 
               dt[filterstep]*(dt[filterstep]*x[4]*x[5]+x[4]*x[3])*np.cos(dt[filterstep]*x[4]+x[2]) - 
               x[3]*np.sin(x[2]) + (dt[filterstep]*x[5] + x[3])*np.sin(dt[filterstep]*x[4]+x[2]))/x[4]**2 - (
                   -x[4]*x[3]*np.sin(x[2]) - x[5]*np.cos(x[2]) +
                   x[5]*np.cos(dt[filterstep]*x[4] + x[2]) + 
                   (dt[filterstep]*x[4]*x[5] + x[4]*x[3])*np.sin(dt[filterstep]*x[4] + x[2])) *2 / x[4]**3
        a16 = (dt[filterstep]*x[4]*np.sin(dt[filterstep]*x[4]+x[2]) - np.cos(x[2]) + np.cos(dt[filterstep]*x[4]+x[2])) / x[4]**2

        a23 = (-x[4]*x[3]*np.sin(x[2]) - x[5]*np.cos(x[2]) + x[5]*np.cos(dt[filterstep]*x[4]+x[2]) -
               (-dt[filterstep]*x[4]*x[5] - x[4]*x[3])*np.sin(dt[filterstep]*x[4]+x[2])) / x[4]**2
        a24 = (x[4]*np.cos(x[2]) - x[4]*np.cos(dt[filterstep]*x[4] + x[2])) / x[4]**2
        a25 = (dt[filterstep]*x[5]*np.cos(dt[filterstep]*x[4] + x[2]) -
               dt[filterstep]*(-dt[filterstep]*x[4]*x[5]-x[4]*x[3])*np.sin(dt[filterstep]*x[4]+x[2]) + 
               x[3]*np.cos(x[2]) + (-dt[filterstep]*x[5]-x[3])*np.cos(dt[filterstep]*x[4]+x[2]))/x[4]**2 - (
                   x[4]*x[3]*np.cos(x[2]) - x[5]*np.sin(x[2]) + 
                   x[5]*np.sin(dt[filterstep]*x[4]+x[2]) +
                   (-dt[filterstep]*x[4]*x[5]-x[4]*x[3])*np.cos(dt[filterstep]*x[4]+x[2])) *2 / x[4]**3
        a26 =  (-dt[filterstep]*x[4]*np.cos(dt[filterstep]*x[4]+x[2]) - np.sin(x[2]) + np.sin(dt[filterstep]*x[4] + x[2])) / x[4]**2
            
        JA = np.matrix([[1.0, 0.0, a13, a14, a15, a16],
                        [0.0, 1.0, a23, a24, a25, a26],
                        [0.0, 0.0, 1.0, 0.0, dt[filterstep], 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, dt[filterstep]],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)

        ## Calculate the Process Noise Covariance Matrix
        s_pos = 0.5*max_acc*dt[filterstep]**2
        s_psi = max_yaw_rate*dt[filterstep]
        s_speed = max_acc*dt[filterstep]
        s_omega = max_yaw_acc*dt[filterstep]
        s_acc = max_jerk*dt[filterstep]

        Q = np.diag([s_pos**2, s_pos**2, s_psi**2, s_speed**2, s_omega**2, s_acc**2])

        ## Project the error covariance ahead
        P = JA*P*JA.T + Q

        ## Measurement Update (Correction)
        hx = np.matrix([[x[3]],[x[4]],[x[5]]])

        JH = np.matrix([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)

        S = JH*P*JH.T + R
        K = (P*JH.T) * np.linalg.inv(S.astype('float'))

        ## Update the estimate
        Z = measurements[:,filterstep].reshape(JH.shape[0],1)
        y = Z - (hx)  ### Innovation or Residual
        x = x + np.array(K*y).reshape(-1)

        ## Limit the speed to be non-negative
        if x[3]<0.:
            x[3] = 0.

        ## Update the error covariance
        P = (I - (K*JH))*P

        ## Save states
        estimates[filterstep,:] = x

    veh[['x_ekf','y_ekf','psi_ekf','v_ekf','omega_ekf','acc_ekf']] = estimates
    if reverse:
        veh = veh.iloc[::-1].reset_index(drop=True)

    return veh


# Reconstruct the trajectory of the surrounding vehicles
# Extended Kalman Filter for Constant Heading and Velocity
# Adapted from https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CHCV.ipynb
def reconstruct_surrounding(veh, params):
    uncertainty_init = params['uncertainty_init']
    uncertainty_pos = params['uncertainty_pos']
    uncertainty_speed = params['uncertainty_speed']
    max_acc = params['max_acc']
    max_yaw_rate = params['max_yaw_rate']
    
    ## Initialize
    numstates = 4
    P = np.eye(numstates)*uncertainty_init # Initial Uncertainty
    dt = np.gradient(veh['time'])
    R = np.diag([uncertainty_pos,uncertainty_pos,uncertainty_speed]) # Measurement Noise
    I = np.eye(numstates)

    ## Measurement vector
    mx, my, mv = veh['x'].values, veh['y'].values, veh['speed_comp'].values
    measurements = np.vstack((mx, my, mv))
    m = measurements.shape[1]

    ## Initial state
    x = np.array([mx[0], my[0], mv[0], 0.])

    ## Estimated vector
    estimates = np.zeros((m,4))

    for filterstep in range(m):
        ## Time Update (Prediction)
        x[0] = x[0] + dt[filterstep]*x[2]*np.cos(x[3])
        x[1] = x[1] + dt[filterstep]*x[2]*np.sin(x[3])
        x[2] = x[2]
        x[3] = (x[3]+ np.pi) % (2.0*np.pi) - np.pi

        ## Calculate the Jacobian of the Dynamic Matrix JA
        a13 = dt[filterstep]*np.cos(x[3])
        a14 = -dt[filterstep]*x[2]*np.sin(x[3])
        a23 = dt[filterstep]*np.sin(x[3])
        a24 = dt[filterstep]*x[2]*np.cos(x[3])
        JA = np.matrix([[1.0, 0.0, a13, a14],
                        [0.0, 1.0, a23, a24],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]], dtype=float)

        ## Calculate the Process Noise Covariance Matrix
        s_pos = 0.5*max_acc*dt[filterstep]**2
        s_psi = max_yaw_rate*dt[filterstep]
        s_speed = max_acc*dt[filterstep]

        Q = np.diag([s_pos**2, s_pos**2, s_speed**2, s_psi**2])

        ## Project the error covariance ahead
        P = JA*P*JA.T + Q

        ## Measurement Update (Correction)
        hx = np.matrix([[x[0]],[x[1]],[x[2]]])

        JH = np.matrix([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]], dtype=float)

        S = JH*P*JH.T + R
        K = (P*JH.T) * np.linalg.inv(S.astype('float'))

        ## Update the estimate
        Z = measurements[:,filterstep].reshape(JH.shape[0],1)
        y = Z - (hx)                         # Innovation or Residual
        x = x + np.array(K*y).reshape(-1)

        ## Limit the speed to be non-negative
        if x[2]<0.:
            x[2] = 0.

        ## Update the error covariance
        P = (I - (K*JH))*P

        ## Save states
        estimates[filterstep,:] = x

    veh[['x_ekf','y_ekf','v_ekf','psi_ekf']] = estimates

    return veh
