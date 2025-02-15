import numpy as np
import warnings
from .geometry_utils import DTC_ij, CurrentD



def TAdv(samples, toreturn='dataframe'):
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


