from .longitudinal_ssms import TTC, DRAC, MTTC, PSD
from .two_dimensional_ssms import TAdv, TTC2D, ACT


# Efficiency evaluation
def evaluate_efficiency(samples, indicator, iterations, average_only=True):
    if indicator=='TTC':
        compute_func = TTC
    elif indicator=='DRAC':
        compute_func = DRAC
    elif indicator=='MTTC':
        compute_func = MTTC
    elif indicator=='PSD':
        compute_func = PSD
    elif indicator=='TAdv':
        compute_func = TAdv
    elif indicator=='TTC2D':
        compute_func = TTC2D
    elif indicator=='ACT':
        compute_func = ACT
    else:
        print('Undefined indicator. Please specify \'TTC\', \'DRAC\', \'MTTC\', \'PSD\', \'TAdv\', \'TTC2D\', or \'ACT\'.')
        return None

    import time as systime
    ts = []
    for _ in range(iterations):
        t = systime.time()
        _ = compute_func(samples, 'values')
        ts.append(systime.time()-t)
    if average_only:
        return sum(ts)/iterations
    else:
        return sum(ts)/iterations, ts
