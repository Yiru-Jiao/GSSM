import numpy as np
import warnings
from .geometry_utils import DTC_ij, CurrentD


def TTC(samples, toreturn='dataframe'):
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        delta_v = np.sqrt((samples['vx_i']-samples['vx_j'])**2+(samples['vy_i']-samples['vy_j'])**2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        ttc_ij = dtc_ij/delta_v
        ttc_ij[leaving_ij<20] = np.inf # inf means the two vehicles will not collide if they keep current velocity
        ttc_ij[(leaving_ij>20)&(leaving_ij%20!=0)] = -1 # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var+'_i' for var in ['x','y','vx','vy','hx','hy','length','width']]
        values = [var+'_j' for var in ['x','y','vx','vy','hx','hy','length','width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        ttc_ji = dtc_ji/delta_v
        ttc_ji[leaving_ji<20] = np.inf
        ttc_ji[(leaving_ji>20)&(leaving_ji%20!=0)] = -1

        if toreturn=='dataframe':
            samples = samples.copy()
            samples['TTC'] = np.minimum(ttc_ij, ttc_ji)
            return samples
        elif toreturn=='values':
            return np.minimum(ttc_ij, ttc_ji).values


def DRAC(samples, toreturn='dataframe'):
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        delta_v = np.sqrt((samples['vx_i']-samples['vx_j'])**2+(samples['vy_i']-samples['vy_j'])**2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        drac_ij = delta_v**2/dtc_ij/2
        drac_ij[leaving_ij<20] = 0. # the two vehicles will not collide if they keep current velocity
        drac_ij[(leaving_ij>20)&(leaving_ij%20!=0)] = -1 # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var+'_i' for var in ['x','y','vx','vy','hx','hy','length','width']]
        values = [var+'_j' for var in ['x','y','vx','vy','hx','hy','length','width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        drac_ji = delta_v**2/dtc_ji/2
        drac_ji[leaving_ji<20] = 0.
        drac_ji[(leaving_ji>20)&(leaving_ji%20!=0)] = -1

        if toreturn=='dataframe':
            samples = samples.copy()
            samples['DRAC'] = np.maximum(drac_ij, drac_ji)
            return samples
        elif toreturn=='values':
            return np.maximum(drac_ij, drac_ji).values


def MTTC(samples, toreturn='dataframe'):
    '''
    https://doi.org/10.3141/2083-12
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    elif 'acc_i' not in samples.columns:
        warnings.warn('Acceleration of the ego vehicle is not provided.')
    else:
        delta_v = np.sqrt((samples['vx_i']-samples['vx_j'])**2+(samples['vy_i']-samples['vy_j'])**2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        ttc_ij = dtc_ij/delta_v
        ttc_ij[leaving_ij<20] = np.inf # inf means the two vehicles will not collide if they keep current velocity
        ttc_ij[(leaving_ij>20)&(leaving_ij%20!=0)] = -1 # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var+'_i' for var in ['x','y','vx','vy','hx','hy','length','width']]
        values = [var+'_j' for var in ['x','y','vx','vy','hx','hy','length','width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        ttc_ji = dtc_ji/delta_v
        ttc_ji[leaving_ji<20] = np.inf
        ttc_ji[(leaving_ji>20)&(leaving_ji%20!=0)] = -1

        ttc = np.minimum(ttc_ij, ttc_ji)
        dtc = np.minimum(dtc_ij, dtc_ji)

        if 'acc_j' in samples.columns:
            acc_i = samples['acc_i'].values
            acc_j = samples['acc_j'].values
            delta_a = acc_i - acc_j
        else: # assume acc_j=0 (i.e., the other vehicle keeps current velocity)
            acc_i = samples['acc_i'].values
            delta_a = acc_i
        delta_v = delta_v*np.sign(((leaving_ij>=20)|(leaving_ji>=20)).astype(int)-0.5) # if the two vehicles are leaving each other, the relative velocity is set negative
        dtc[(abs(delta_a)<1e-6)&np.isinf(dtc)] = 1e15 # to avoid multiplication error when delta_a=0 and dtc=inf
        delta_a[(delta_a>0)&(delta_a<1e-6)] = 1e-7
        delta_a[(delta_a<0)&(delta_a>-1e-6)] = -1e-7
        squared_term = delta_v**2 + 2*delta_a*dtc
        squared_term[squared_term>=0] = np.sqrt(squared_term[squared_term>=0])
        squared_term[squared_term<0] = np.nan
        mttc_plus = (-delta_v + squared_term) / delta_a
        mttc_minus = (-delta_v - squared_term) / delta_a
        mttc = mttc_minus.copy()
        mttc[(mttc_minus<=0)&(mttc_plus>0)] = mttc_plus[(mttc_minus<=0)&(mttc_plus>0)]
        mttc[(mttc_minus<=0)&(mttc_plus<=0)] = np.inf
        mttc[(np.isnan(mttc_minus)|np.isnan(mttc_plus))] = np.inf
        mttc[abs(delta_a)<1e-6] = ttc[abs(delta_a)<1e-6]
        mttc[((leaving_ij>20)&(leaving_ij%20!=0))|((leaving_ji>20)&(leaving_ji%20!=0))] = -1

        if toreturn=='dataframe':
            samples = samples.copy()
            samples['MTTC'] = mttc
            return samples
        elif toreturn=='values':
            return mttc.values
        

def PSD(samples, toreturn='dataframe', braking_dec=5.5):
    '''
    https://onlinepubs.trb.org/Onlinepubs/trr/1978/667/667-009.pdf
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        v_ego = np.sqrt(samples['vx_i']**2+samples['vy_i']**2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        braking_dist = v_ego**2 / 2 / braking_dec
        psd = dtc_ij / braking_dist
        psd[leaving_ij<20] = 10. # the two vehicles will not collide if they keep current velocity
        psd[(leaving_ij>20)&(leaving_ij%20!=0)] = -1 # -1 means the bounding boxes of the two vehicles are overlapping

        if toreturn=='dataframe':
            samples = samples.copy()
            samples['PSD'] = psd
            return samples
        elif toreturn=='values':
            return psd.values


def TTC_DRAC_MTTC(samples, toreturn='dataframe'):
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    elif 'acc_i' not in samples.columns:
        warnings.warn('Acceleration of the ego vehicle is not provided.')
    else:
        delta_v = np.sqrt((samples['vx_i']-samples['vx_j'])**2+(samples['vy_i']-samples['vy_j'])**2)
        dtc_ij, leaving_ij = DTC_ij(samples)
        ttc_ij = dtc_ij/delta_v
        ttc_ij[leaving_ij<20] = np.inf # inf means the two vehicles will not collide if they keep current velocity
        ttc_ij[(leaving_ij>20)&(leaving_ij%20!=0)] = -1 # -1 means the bounding boxes of the two vehicles are overlapping
        drac_ij = delta_v**2/dtc_ij/2
        drac_ij[leaving_ij<20] = 0. # the two vehicles will not collide if they keep current velocity
        drac_ij[(leaving_ij>20)&(leaving_ij%20!=0)] = -1 # -1 means the bounding boxes of the two vehicles are overlapping

        keys = [var+'_i' for var in ['x','y','vx','vy','hx','hy','length','width']]
        values = [var+'_j' for var in ['x','y','vx','vy','hx','hy','length','width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
        ttc_ji = dtc_ji/delta_v
        ttc_ji[leaving_ji<20] = np.inf
        ttc_ji[(leaving_ji>20)&(leaving_ji%20!=0)] = -1
        drac_ji = delta_v**2/dtc_ji/2
        drac_ji[leaving_ji<20] = 0.
        drac_ji[(leaving_ji>20)&(leaving_ji%20!=0)] = -1

        dtc = np.minimum(dtc_ij, dtc_ji)
        ttc = np.minimum(ttc_ij, ttc_ji)
        drac = np.maximum(drac_ij, drac_ji)

        if 'acc_j' in samples.columns:
            acc_i = samples['acc_i'].values
            acc_j = samples['acc_j'].values
            delta_a = acc_i - acc_j
        else: # assume acc_j=0 (i.e., the other vehicle keeps current velocity)
            acc_i = samples['acc_i'].values
            delta_a = acc_i
        delta_v = delta_v*np.sign(((leaving_ij>=20)|(leaving_ji>=20)).astype(int)-0.5) # if the two vehicles are leaving each other, the relative velocity is set negative
        squared_term = delta_v**2 + 2*delta_a*dtc
        squared_term[squared_term>=0] = np.sqrt(squared_term[squared_term>=0])
        squared_term[squared_term<0] = np.nan
        mttc_plus = (-delta_v + squared_term) / delta_a
        mttc_minus = (-delta_v - squared_term) / delta_a
        mttc = mttc_minus.copy()
        mttc[(mttc_minus<=0)&(mttc_plus>0)] = mttc_plus[(mttc_minus<=0)&(mttc_plus>0)]
        mttc[(mttc_minus<=0)&(mttc_plus<=0)] = np.inf
        mttc[(np.isnan(mttc_minus)|np.isnan(mttc_plus))] = np.inf
        mttc[abs(delta_a)<1e-6] = ttc[abs(delta_a)<1e-6]
        mttc[((leaving_ij>20)&(leaving_ij%20!=0))|((leaving_ji>20)&(leaving_ji%20!=0))] = -1

        if toreturn=='dataframe':
            samples = samples.copy()
            samples['TTC'] = ttc
            samples['DRAC'] = drac
            samples['MTTC'] = mttc
            return samples
        elif toreturn=='values':
            return ttc.values, drac.values, mttc.values

