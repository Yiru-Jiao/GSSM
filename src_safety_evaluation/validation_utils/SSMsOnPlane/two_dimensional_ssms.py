import numpy as np
import pandas as pd
import warnings
from .geometry_utils import DTC_ij, CurrentD, rotate_coor, getpoints



def TAdv(samples, toreturn='dataframe'):
    '''
    https://doi.org/10.1016/j.aap.2010.03.021
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        dtc_ij, leaving_ij = DTC_ij(samples)
        t_ij = dtc_ij/np.sqrt(samples['vx_i']**2+samples['vy_i']**2)

        keys = [var+'_i' for var in ['x','y','vx','vy','hx','hy','length','width']]
        values = [var+'_j' for var in ['x','y','vx','vy','hx','hy','length','width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        t_ji = dtc_ji/np.sqrt(samples['vx_j']**2+samples['vy_j']**2)

        time_advantage = abs(t_ij - t_ji)
        time_advantage[(leaving_ij<20)&(leaving_ji<20)] = np.inf # the two vehicles will not collide if they keep current velocity
        time_advantage[((leaving_ij>20)&(leaving_ij%20!=0))|((leaving_ji>20)&(leaving_ji%20!=0))] = -1 # the bounding boxes of the two vehicles are overlapping

        if toreturn=='dataframe':
            samples['TAdv'] = time_advantage
            return samples
        elif toreturn=='values':
            return time_advantage


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
        samples = samples.reset_index(drop=True)
        # get front center points of vehicles i and j
        front_i, _, front_j, _ = getpoints(samples, front_rear_only=True)
        samples['frontx_i'], samples['fronty_i'] = front_i[0], front_i[1]
        samples['frontx_j'], samples['fronty_j'] = front_j[0], front_j[1]

        # determine leading/following vehicles
        ## consider i as the following vehicle
        x_lon_axis = samples['hx_i'].values
        y_lon_axis = samples['hy_i'].values
        _, yi_j = rotate_coor(x_lon_axis, y_lon_axis, samples['x_j'].values, samples['y_j'].values)
        ## if yi_j<0, j is the following vehicle
        j_following = yi_j<0
        ## divide samples into two groups
        samples_i_following = samples[~j_following]
        samples_j_following = samples[j_following]

        # calculate 2D-TTC for each group
        ttc_lon_i_following, ttc_lat_i_following = get_ttc_components(samples_i_following, following='i')
        samples_i_following['TTC2D'] = np.minimum(ttc_lon_i_following, ttc_lat_i_following)
        ttc_lon_j_following, ttc_lat_j_following = get_ttc_components(samples_j_following, following='j')
        samples_j_following['TTC2D'] = np.minimum(ttc_lon_j_following, ttc_lat_j_following)

        # merge the two groups
        samples = pd.concat([samples_i_following, samples_j_following], axis=0).sort_index()

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
        samples = samples.reset_index(drop=True)
        samples['original_index'] = samples.index.values
        samples = samples.sort_values(by=['target_id','time']).set_index('target_id')
        # for each event
        for target_id in target_ids:
            event = samples.loc[target_id]
            delta = DTC_ij(event, 'values')
            pdelta_pt = np.gradient(delta, event['time'].values)
            pdelta_pt[(pdelta_pt>=0)&(pdelta_pt<1e-6)] = 1e-6
            pdelta_pt[(pdelta_pt<0)&(pdelta_pt>-1e-6)] = -1e-6
            samples.loc[target_id, 'ACT'] = delta / pdelta_pt
        samples.loc[samples['ACT']<=0, 'ACT'] = np.inf
        samples = samples.sort_values(by='original_index').reset_index().drop(columns=['original_index'])

        if toreturn=='dataframe':
            return samples
        elif toreturn=='values':
            return samples['ACT'].values
