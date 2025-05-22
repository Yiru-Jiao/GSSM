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
        delta_t = dt[filterstep]
        if abs(x[4]) < 1e-3: ## Driving straight
            delta_x = np.cos(x[2]) * (x[3]*delta_t + 0.5*x[5]*delta_t**2)
            delta_y = np.sin(x[2]) * (x[3]*delta_t + 0.5*x[5]*delta_t**2)
            ## Elements in the Jacobian of the Dynamic Matrix
            a13 = -np.sin(x[2]) * (x[3]*delta_t + 0.5*x[5]*delta_t**2)
            a14 = np.cos(x[2]) * delta_t
            a15 = 0.
            a16 = np.cos(x[2]) * 0.5*delta_t**2
            a23 = np.cos(x[2]) * (x[3]*delta_t + 0.5*x[5]*delta_t**2)
            a24 = np.sin(x[2]) * delta_t
            a25 = 0.
            a26 = np.sin(x[2]) * 0.5*delta_t**2
        else:
            delta_x = (1/x[4]**2) * (-x[3]*x[4]*np.sin(x[2]) -x[5]*np.cos(x[2]) +
                                     x[5]*np.cos(x[2]+x[4]*delta_t) + 
                                     (x[5]*x[4]*delta_t+x[3]*x[4])*np.sin(x[2]+x[4]*delta_t))
            delta_y = (1/x[4]**2) * (x[3]*x[4]*np.cos(x[2]) - x[5]*np.sin(x[2]) +
                                     x[5]*np.sin(x[2]+x[4]*delta_t) +
                                     (-x[5]*x[4]*delta_t-x[3]*x[4])*np.cos(x[2]+x[4]*delta_t))
            ## Elements in the Jacobian of the Dynamic Matrix
            a13 = (-x[4]*x[3]*np.cos(x[2]) + x[5]*np.sin(x[2]) - x[5]*np.sin(delta_t*x[4]+x[2]) +
                (delta_t*x[4]*x[5]+x[4]*x[3])*np.cos(delta_t*x[4]+x[2])) / x[4]**2
            a14 = (-x[4]*np.sin(x[2]) + x[4]*np.sin(delta_t*x[4]+x[2])) / x[4]**2
            a15 = (-delta_t*x[5]*np.sin(delta_t*x[4]+x[2]) + 
                delta_t*(delta_t*x[4]*x[5]+x[4]*x[3])*np.cos(delta_t*x[4]+x[2]) - 
                x[3]*np.sin(x[2]) + (delta_t*x[5] + x[3])*np.sin(delta_t*x[4]+x[2]))/x[4]**2 - (
                    -x[4]*x[3]*np.sin(x[2]) - x[5]*np.cos(x[2]) +
                    x[5]*np.cos(delta_t*x[4] + x[2]) + 
                    (delta_t*x[4]*x[5] + x[4]*x[3])*np.sin(delta_t*x[4] + x[2])) *2 / x[4]**3
            a16 = (delta_t*x[4]*np.sin(delta_t*x[4]+x[2]) - np.cos(x[2]) + np.cos(delta_t*x[4]+x[2])) / x[4]**2

            a23 = (-x[4]*x[3]*np.sin(x[2]) - x[5]*np.cos(x[2]) + x[5]*np.cos(delta_t*x[4]+x[2]) -
                (-delta_t*x[4]*x[5] - x[4]*x[3])*np.sin(delta_t*x[4]+x[2])) / x[4]**2
            a24 = (x[4]*np.cos(x[2]) - x[4]*np.cos(delta_t*x[4] + x[2])) / x[4]**2
            a25 = (delta_t*x[5]*np.cos(delta_t*x[4] + x[2]) -
                delta_t*(-delta_t*x[4]*x[5]-x[4]*x[3])*np.sin(delta_t*x[4]+x[2]) + 
                x[3]*np.cos(x[2]) + (-delta_t*x[5]-x[3])*np.cos(delta_t*x[4]+x[2]))/x[4]**2 - (
                    x[4]*x[3]*np.cos(x[2]) - x[5]*np.sin(x[2]) + 
                    x[5]*np.sin(delta_t*x[4]+x[2]) +
                    (-delta_t*x[4]*x[5]-x[4]*x[3])*np.cos(delta_t*x[4]+x[2])) *2 / x[4]**3
            a26 =  (-delta_t*x[4]*np.cos(delta_t*x[4]+x[2]) - np.sin(x[2]) + np.sin(delta_t*x[4] + x[2])) / x[4]**2

        x[0] = x[0] + delta_x
        x[1] = x[1] + delta_y
        x[2] = (x[2] + x[4] * delta_t + np.pi) % (2.0 * np.pi) - np.pi
        x[3] = x[3] + x[5] * delta_t
        x[4] = x[4]
        x[5] = x[5]
            
        JA = np.matrix([[1.0, 0.0, a13, a14, a15, a16],
                        [0.0, 1.0, a23, a24, a25, a26],
                        [0.0, 0.0, 1.0, 0.0, delta_t, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, delta_t],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)

        ## Calculate the Process Noise Covariance Matrix
        s_pos = 0.5*max_acc*delta_t**2
        s_psi = max_yaw_rate*delta_t
        s_speed = max_acc*delta_t
        s_omega = max_yaw_acc*delta_t
        s_acc = max_jerk*delta_t

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
        delta_t = dt[filterstep]
        x[0] = x[0] + delta_t*x[2]*np.cos(x[3])
        x[1] = x[1] + delta_t*x[2]*np.sin(x[3])
        x[2] = x[2]
        x[3] = (x[3]+ np.pi) % (2.0*np.pi) - np.pi

        ## Calculate the Jacobian of the Dynamic Matrix JA
        a13 = delta_t*np.cos(x[3])
        a14 = -delta_t*x[2]*np.sin(x[3])
        a23 = delta_t*np.sin(x[3])
        a24 = delta_t*x[2]*np.cos(x[3])
        JA = np.matrix([[1.0, 0.0, a13, a14],
                        [0.0, 1.0, a23, a24],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]], dtype=float)

        ## Calculate the Process Noise Covariance Matrix
        s_pos = 0.5*max_acc*delta_t**2
        s_psi = max_yaw_rate*delta_t
        s_speed = max_acc*delta_t

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
