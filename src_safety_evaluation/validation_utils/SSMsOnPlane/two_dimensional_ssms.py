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
        # determine leading/following vehicles in case of parallel velocities
        ## consider i as an ego vehicle
        x_lon_axis = samples['hx_i'].values
        y_lon_axis = samples['hy_i'].values
        _, yi_j = rotate_coor(x_lon_axis, y_lon_axis, (samples['x_j']-samples['x_i']).values, (samples['y_j']-samples['y_i']).values)
        _, vyi_j = rotate_coor(x_lon_axis, y_lon_axis, samples['vx_j'].values, samples['vy_j'].values)
        ## if yi_j<0, j is behind i; if vyi_j<0, v_j is opposite to v_i
        speed_i = np.sqrt(samples['vx_i']**2+samples['vy_i']**2).values
        speed_j = np.sqrt(samples['vx_j']**2+samples['vy_j']**2).values
        relative_v = speed_i - speed_j # by default, consider i follows j and they move in the same direction
        j_following_i_same_direction = (yi_j<0)&(vyi_j>=0)
        relative_v[j_following_i_same_direction] = (speed_j - speed_i)[j_following_i_same_direction]
        opposite_direction = (vyi_j<0)
        relative_v[opposite_direction] = (speed_i + speed_j)[opposite_direction]
        ## avoid zero division
        relative_v[(relative_v>=0)&(relative_v<1e-6)] = 1e-6
        relative_v[(relative_v<0)&(relative_v>-1e-6)] = -1e-6
        # precompute ttc in case of parallel velocities
        current_distance = CurrentD(samples, toreturn='values')
        ttc = current_distance / relative_v
        # if ttc<0, the two vehicles leave each other and set ttc to infinity
        ttc[ttc<0] = np.inf

        tadv_mat = []
        # For each point of vehicle i
        for point_i in [point_i1, point_i2, point_i3, point_i4]:
            # For each point of vehicle j
            for point_j in [point_j1, point_j2, point_j3, point_j4]:
                # intersection point between 
                # 1) the line extended from the point of vehicle i along the direction of its velocity and
                # 2) the line extended from the point of vehicle j along the direction of its velocity
                ist = intersect(line(point_i, point_i+v_i), line(point_j, point_j+v_j))
                # distance from the intersection point to both points of vehicle i and j
                dist_ist_i = np.sqrt((ist[0]-point_i[0])**2+(ist[1]-point_i[1])**2)
                dist_ist_j = np.sqrt((ist[0]-point_j[0])**2+(ist[1]-point_j[1])**2)
                # time advantage is the time difference between the two vehicles reaching the intersection point
                predicted_time_i = dist_ist_i / np.maximum(np.sqrt(v_i[0]**2+v_i[1]**2), 1e-6)
                predicted_time_j = dist_ist_j / np.maximum(np.sqrt(v_j[0]**2+v_j[1]**2), 1e-6)
                time_advantage = np.absolute(predicted_time_i - predicted_time_j)
                # if the two lines are parallel (threshold: 3 degrees), time advantage equals to TTC
                angle_ij = angle(v_i[0], v_i[1], v_j[0], v_j[1]) # [-pi, pi]
                parallel_lines = np.isnan(ist[0])|(abs(angle_ij)<(np.pi/60))|(abs(angle_ij)>(np.pi*59/60))
                time_advantage[parallel_lines] = ttc[parallel_lines]
                # for unparallel cases, if the intersection point is not ahead of both vehicles, set time advantage to infinity
                ist_ahead_i = ((ist[0]-point_i[0])*v_i[0]+(ist[1]-point_i[1])*v_i[1]) >= 0
                ist_ahead_j = ((ist[0]-point_j[0])*v_j[0]+(ist[1]-point_j[1])*v_j[1]) >= 0
                time_advantage[(~parallel_lines)&(~(ist_ahead_i&ist_ahead_j))] = np.inf
                # for parallel cases, cases when ttc<0 has already been set to infinity
                # append the time advantage
                tadv_mat.append(time_advantage)

        time_advantage = np.array(tadv_mat).min(axis=0)

        if toreturn=='dataframe':
            samples = samples.copy()
            samples['TAdv'] = time_advantage
            return samples
        elif toreturn=='values':
            return time_advantage


def get_ttc_components(samples, following='i'):
    if following=='i':
        leading = 'j'
    elif following=='j':
        leading = 'i'
    
    x_following_front, y_following_front = rotate_coor(samples['hx_'+following].values,
                                                       samples['hy_'+following].values,
                                                       samples['frontx_'+following].values,
                                                       samples['fronty_'+following].values)
    x_leading_front, y_leading_front = rotate_coor(samples['hx_'+following].values,
                                                   samples['hy_'+following].values,
                                                   samples['frontx_'+leading].values,
                                                   samples['fronty_'+leading].values)
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
        ## consider i as an ego vehicle
        x_lon_axis = samples['hx_i'].values
        y_lon_axis = samples['hy_i'].values
        _, yi_j = rotate_coor(x_lon_axis, y_lon_axis, (samples['x_j']-samples['x_i']).values, (samples['y_j']-samples['y_i']).values)
        ## if yi_j<0, j is behind i
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
    We don't use acceleration or yaw rate for vehicle dynamics because these two variables are not available in SHRP2 data.
    This means that this computation assumes acceleration=0 and yaw rate=0 for both vehicles.
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
        num_samples = len(samples)

        dist_mat = []
        point_p_ij = []
        point_p_x = []
        point_p_y = []
        point_delta_x = []
        point_delta_y = []
        # for each edge (and point) of vehicle i
        for point_i_start, point_i_end in zip([point_i1, point_i4, point_i3, point_i2],
                                              [point_i2, point_i3, point_i1, point_i4]):
            # for each edge (and point) of vehicle j
            for point_j_start, point_j_end in zip([point_j1, point_j4, point_j3, point_j2],
                                                  [point_j2, point_j3, point_j1, point_j4]):
                # find the shortest distance from the point of vehicle i to the edge of vehicle j and determine point_delta
                point_p_ij.append(np.zeros(num_samples))
                point_p_x.append(point_i_start[0])
                point_p_y.append(point_i_start[1])
                ## distance from the point of i to the end points of the edge of j
                dist_i_j1 = np.sqrt((point_i_start[0]-point_j_start[0])**2+(point_i_start[1]-point_j_start[1])**2)
                dist_i_j2 = np.sqrt((point_i_start[0]-point_j_end[0])**2+(point_i_start[1]-point_j_end[1])**2)
                ## phi_1 and phi_2 as the bottom angles of the triangle formed by the point and edge
                phi_1 = angle(point_j_start[0]-point_j_end[0],
                              point_j_start[1]-point_j_end[1],
                              point_i_start[0]-point_j_end[0],
                              point_i_start[1]-point_j_end[1])
                phi_2 = angle(point_j_end[0]-point_j_start[0],
                              point_j_end[1]-point_j_start[1],
                              point_i_start[0]-point_j_start[0],
                              point_i_start[1]-point_j_start[1])
                ## consider the closest point as one of the end points
                closest_dist = dist_i_j1.copy()
                p_delta_x = point_j_start[0].copy()
                p_delta_y = point_j_start[1].copy()
                end_closer = dist_i_j2 < dist_i_j1
                closest_dist[end_closer] = dist_i_j2[end_closer]
                p_delta_x[end_closer] = point_j_end[0][end_closer]
                p_delta_y[end_closer] = point_j_end[1][end_closer]
                ## if the absolute values of phi_1 and phi_2 are both less than pi/2, the closest point is on the edge
                on_edge = (abs(phi_1)<np.pi/2)&(abs(phi_2)<np.pi/2)
                ist = intersect(line(point_i_start, point_i_start+np.array([-(point_j_start-point_j_end)[1],(point_j_start-point_j_end)[0]])), 
                                line(point_j_start, point_j_end))
                dist_p_edge = np.sqrt((ist[0]-point_i_start[0])**2+(ist[1]-point_i_start[1])**2)
                closest_dist[on_edge] = dist_p_edge[on_edge]
                p_delta_x[on_edge] = ist[0][on_edge]
                p_delta_y[on_edge] = ist[1][on_edge]
                ## append the distance and the point_delta at the closest distance
                dist_mat.append(closest_dist)
                point_delta_x.append(p_delta_x)
                point_delta_y.append(p_delta_y)

                # find the shortest distance from the point of vehicle j to the edge of vehicle i and determine point_delta
                point_p_ij.append(np.ones(num_samples))
                point_p_x.append(point_j_start[0])
                point_p_y.append(point_j_start[1])
                ## distance from the point of j to the end points of the edge of i
                dist_j_i1 = np.sqrt((point_j_start[0]-point_i_start[0])**2+(point_j_start[1]-point_i_start[1])**2)
                dist_j_i2 = np.sqrt((point_j_start[0]-point_i_end[0])**2+(point_j_start[1]-point_i_end[1])**2)
                ## phi_1 and phi_2 as the bottom angles of the triangle formed by the point and edge
                phi_1 = angle(point_i_start[0]-point_i_end[0],
                              point_i_start[1]-point_i_end[1],
                              point_j_start[0]-point_i_end[0],
                              point_j_start[1]-point_i_end[1])
                phi_2 = angle(point_i_end[0]-point_i_start[0],
                              point_i_end[1]-point_i_start[1],
                              point_j_start[0]-point_i_start[0],
                              point_j_start[1]-point_i_start[1])
                ## consider the closest point as one of the end points
                closest_dist = dist_j_i1.copy()
                p_delta_x = point_i_start[0].copy()
                p_delta_y = point_i_start[1].copy()
                end_closer = dist_j_i2 < dist_j_i1
                closest_dist[end_closer] = dist_j_i2[end_closer]
                p_delta_x[end_closer] = point_i_end[0][end_closer]
                p_delta_y[end_closer] = point_i_end[1][end_closer]
                ## if the absolute values of phi_1 and phi_2 are both less than pi/2, the closest point is on the edge
                on_edge = (abs(phi_1)<np.pi/2)&(abs(phi_2)<np.pi/2)
                ist = intersect(line(point_j_start, point_j_start+np.array([-(point_i_start-point_i_end)[1],(point_i_start-point_i_end)[0]])), 
                                line(point_i_start, point_i_end))
                dist_p_edge = np.sqrt((ist[0]-point_j_start[0])**2+(ist[1]-point_j_start[1])**2)
                closest_dist[on_edge] = dist_p_edge[on_edge]
                p_delta_x[on_edge] = ist[0][on_edge]
                p_delta_y[on_edge] = ist[1][on_edge]
                ## append the distance and the point_delta at the closest distance
                dist_mat.append(closest_dist)
                point_delta_x.append(p_delta_x)
                point_delta_y.append(p_delta_y)

        # across all points and edges, find the minimum distance
        dist_mat = np.array(dist_mat)
        dist_mat[np.isnan(dist_mat)] = np.inf
        min_index = np.argmin(np.array(dist_mat), axis=0)
        shortest_dist = dist_mat[min_index, np.arange(len(min_index))]
        point_p_ij = np.array(point_p_ij)[min_index, np.arange(len(min_index))] # 0: point i, 1: point j
        point_p_x = np.array(point_p_x)[min_index, np.arange(len(min_index))]
        point_p_y = np.array(point_p_y)[min_index, np.arange(len(min_index))]
        point_delta_x = np.array(point_delta_x)[min_index, np.arange(len(min_index))]
        point_delta_y = np.array(point_delta_y)[min_index, np.arange(len(min_index))]

        # get the components of velocity in the direction of the shortest distance, i.e., p->delta or delta->p
        v_i = samples[['vx_i','vy_i']].values.T
        v_j = samples[['vx_j','vy_j']].values.T
        from_i_to_j_x = (point_p_x-point_delta_x)*point_p_ij + (point_delta_x-point_p_x)*(1-point_p_ij)
        from_i_to_j_y = (point_p_y-point_delta_y)*point_p_ij + (point_delta_y-point_p_y)*(1-point_p_ij)
        from_j_to_i_x = (point_delta_x-point_p_x)*point_p_ij + (point_p_x-point_delta_x)*(1-point_p_ij)
        from_j_to_i_y = (point_delta_y-point_p_y)*point_p_ij + (point_p_y-point_delta_y)*(1-point_p_ij)
        v_ij = (v_i[0]*from_i_to_j_x + v_i[1]*from_i_to_j_y) / np.sqrt(from_i_to_j_x**2 + from_i_to_j_y**2)
        v_ji = (v_j[0]*from_j_to_i_x + v_j[1]*from_j_to_i_y) / np.sqrt(from_j_to_i_x**2 + from_j_to_i_y**2)

        # calculate p_delta_p_t
        p_delta_pt = v_ij + v_ji
        p_delta_pt[(p_delta_pt>=0)&(p_delta_pt<1e-6)] = 1e-6
        p_delta_pt[(p_delta_pt<0)&(p_delta_pt>-1e-6)] = -1e-6

        # calculate ACT
        act = shortest_dist / p_delta_pt
        act[act<0] = np.inf

        if toreturn=='dataframe':
            samples = samples.copy()
            samples['ACT'] = act
            return samples
        elif toreturn=='values':
            return act
