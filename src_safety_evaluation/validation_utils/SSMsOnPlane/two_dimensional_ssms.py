import numpy as np
import pandas as pd
import warnings
from .geometry_utils import *


def TAdv(samples, toreturn='dataframe'):
    '''
    https://doi.org/10.1016/j.aap.2010.03.021
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
        v_i = samples[['vx_i','vy_i']].values.T
        v_j = samples[['vx_j','vy_j']].values.T
        relative_v = v_i - v_j

        tadv_mat = []
        # For each point of vehicle i
        for point_i, point_other_i in zip([point_i1, point_i2, point_i3, point_i4], [point_i2, point_i1, point_i4, point_i3]):
            # For each point of vehicle j
            for point_j, point_other_j in zip([point_j1, point_j2, point_j3, point_j4], [point_j2, point_j1, point_j4, point_j3]):
                # intersection point between 
                # 1) the line extended from the point of vehicle i along the direction of its velocity and
                # 2) the line extended from the point of vehicle j along the direction of its velocity
                ist = intersect(line(point_i, point_i+v_i), line(point_j, point_j+v_j))
                # distance from the intersection point to both points of vehicle i and j
                dist_ist_i = np.sqrt((ist[0]-point_i[0])**2+(ist[1]-point_i[1])**2)
                dist_ist_j = np.sqrt((ist[0]-point_j[0])**2+(ist[1]-point_j[1])**2)
                # time advantage is the time difference between the two vehicles reaching the intersection point
                predicted_time_i = dist_ist_i / np.minimum(np.sqrt(v_i[0]**2+v_i[1]**2), 1e-6)
                predicted_time_j = dist_ist_j / np.minimum(np.sqrt(v_j[0]**2+v_j[1]**2), 1e-6)
                time_advantage = np.absolute(predicted_time_i - predicted_time_j)
                # if the two lines are parallel, time advantage equals to TTC
                parallel_lines = np.isnan(ist[0])
                dist_ist_i[parallel_lines] = dist_p2l(point_i, point_j, point_other_j)[parallel_lines]
                dist_ist_j[parallel_lines] = dist_p2l(point_j, point_i, point_other_i)[parallel_lines]
                ttc_i = dist_ist_i / np.minimum(np.sqrt(relative_v[0]**2+relative_v[1]**2), 1e-6)
                ttc_j = dist_ist_j / np.minimum(np.sqrt(relative_v[0]**2+relative_v[1]**2), 1e-6)
                time_advantage[parallel_lines] = np.minimum(ttc_i, ttc_j)[parallel_lines]
                # if the intersection point is not ahead of both vehicles, set time advantage to infinity
                ist_ahead_i = np.dot(ist-point_i, v_i)>0
                ist_ahead_j = np.dot(ist-point_j, v_j)>0
                time_advantage[(~parallel_lines)&(~(ist_ahead_i&ist_ahead_j))] = np.inf
                ist_ahead_i[parallel_lines] = (np.dot(point_j-point_i, v_i)>0)[parallel_lines]
                ist_ahead_j[parallel_lines] = (np.dot(point_i-point_j, v_j)>0)[parallel_lines]
                time_advantage[parallel_lines&(~(ist_ahead_i|ist_ahead_j))] = np.inf
                # append the time advantage
                tadv_mat.append(time_advantage)

        time_advantage = np.array(tadv_mat).min(axis=0)

        if toreturn=='dataframe':
            samples['TAdv'] = time_advantage
            return samples
        elif toreturn=='values':
            return time_advantage.values


def get_ttc_components(samples, following='i'):
    if following=='i':
        leading = 'j'
    elif following=='j':
        leading = 'i'
    
    x_leading_front, y_leading_front = rotate_coor(samples['hx_'+following].values,
                                                   samples['hy_'+following].values,
                                                   samples['frontx_'+leading].values,
                                                   samples['fronty_'+leading].values)
    x_following_front, y_following_front = rotate_coor(samples['hx_'+following].values,
                                                       samples['hy_'+following].values,
                                                       samples['frontx_'+following].values,
                                                       samples['fronty_'+following].values)
    s0_lat = x_leading_front - x_following_front
    s0_lon = y_leading_front - y_following_front
    l_leading = samples['length_'+leading].values
    width = (samples['width_'+leading].values + samples['width_'+following].values) / 2
    v0_lat, v0_lon = rotate_coor(samples['hx_'+following].values, # the velocity of the leading vehicle
                                 samples['hy_'+following].values,
                                 samples['vx_'+leading].values,
                                 samples['vy_'+leading].values)
    v_lat, v_lon = rotate_coor(samples['hx_'+following].values, # the velocity of the following vehicle
                               samples['hy_'+following].values,
                               samples['vx_'+following].values,
                               samples['vy_'+following].values)
    
    delta_v_lon = v_lon - v0_lon
    delta_v_lon[(delta_v_lon>=0)&(delta_v_lon<1e-6)] = 1e-6
    delta_v_lon[(delta_v_lon<0)&(delta_v_lon>-1e-6)] = -1e-6
    ttc_lon = (s0_lon - l_leading) / delta_v_lon
    condition1 = (s0_lon > l_leading)
    condition2 = (v_lon > v0_lon)
    condition3 = (s0_lat - (v_lat - v0_lat) * ttc_lon) < width
    ttc_lon[(~condition1)|(~condition2)|(~condition3)] = np.inf

    delta_v_lat = v_lat - v0_lat
    delta_v_lat[(delta_v_lat>=0)&(delta_v_lat<1e-6)] = 1e-6
    delta_v_lat[(delta_v_lat<0)&(delta_v_lat>-1e-6)] = -1e-6
    ttc_lat = (s0_lat - width) / delta_v_lat
    condition1 = (s0_lat > width)
    condition2 = (v_lat > v0_lat)
    condition3 = (s0_lon - (v_lon - v0_lon) * ttc_lat) < l_leading
    ttc_lat[(~condition1)|(~condition2)|(~condition3)] = np.inf

    return ttc_lon, ttc_lat


def TTC2D(samples, toreturn='dataframe'):
    '''
    https://doi.org/10.1016/j.aap.2023.107063
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        original_indices = samples.index.values
        samples = samples.reset_index(drop=True)
        # get front center points of vehicles i and j
        front_i, _, front_j, _ = getpoints(samples, front_rear_only=True)
        samples['frontx_i'], samples['fronty_i'] = front_i[:,0], front_i[:,1]
        samples['frontx_j'], samples['fronty_j'] = front_j[:,0], front_j[:,1]

        # determine leading/following vehicles
        ## consider i as the following vehicle
        x_lon_axis = samples['hx_i'].values
        y_lon_axis = samples['hy_i'].values
        _, yi_j = rotate_coor(x_lon_axis, y_lon_axis, samples['x_j'].values, samples['y_j'].values)
        ## if yi_j<0, j is the following vehicle
        j_following = yi_j<0
        ## divide samples into two groups
        samples_i_following = samples[~j_following].copy()
        samples_j_following = samples[j_following].copy()

        # calculate 2D-TTC for each group
        ttc_lon_i_following, ttc_lat_i_following = get_ttc_components(samples_i_following, following='i')
        samples_i_following['TTC2D'] = np.minimum(ttc_lon_i_following, ttc_lat_i_following)
        ttc_lon_j_following, ttc_lat_j_following = get_ttc_components(samples_j_following, following='j')
        samples_j_following['TTC2D'] = np.minimum(ttc_lon_j_following, ttc_lat_j_following)

        # merge the two groups
        samples = pd.concat([samples_i_following, samples_j_following], axis=0).sort_index()
        samples = samples.set_index(original_indices)

        if toreturn=='dataframe':
            return samples
        elif toreturn=='values':
            return samples['TTC2D'].values


def ACT(samples, toreturn='dataframe'):
    '''
    https://doi.org/10.1016/j.trc.2022.103655
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    elif 'time' not in samples.columns:
        warnings.warn('Time reference not found but required for ACT calculation.')
    else:
        target_ids = samples['target_id'].unique()
        original_indices = samples.index.values
        samples = samples.reset_index(drop=True)
        samples['original_index'] = samples.index.values
        samples = samples.sort_values(by=['target_id','time']).set_index('target_id')
        # for each event
        for target_id in target_ids:
            event = samples.loc[target_id]
            delta = CurrentD(event, toreturn='values')
            pdelta_pt = np.gradient(delta, -event['time'].values)
            pdelta_pt[(pdelta_pt>=0)&(pdelta_pt<1e-6)] = 1e-6
            pdelta_pt[(pdelta_pt<0)&(pdelta_pt>-1e-6)] = -1e-6
            samples.loc[target_id, 'ACT'] = delta / pdelta_pt
        samples.loc[samples['ACT']<=0, 'ACT'] = np.inf
        samples = samples.reset_index().sort_values(by='original_index').drop(columns=['original_index'])
        samples = samples.set_index(original_indices)

        if toreturn=='dataframe':
            return samples
        elif toreturn=='values':
            return samples['ACT'].values
