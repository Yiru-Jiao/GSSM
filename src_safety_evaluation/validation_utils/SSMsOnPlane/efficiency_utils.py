import numpy as np
import warnings
from .longitudinal_ssms import TTC, DRAC, MTTC, TTC_DRAC_MTTC
from .two_dimensional_ssms import TAdv



# Efficiency evaluation
def efficiency(samples, indicator, iterations):
    if indicator=='TTC':
        compute_func = TTC
    elif indicator=='DRAC':
        compute_func = DRAC
    elif indicator=='MTTC':
        compute_func = MTTC
    elif indicator=='TTC_DRAC_MTTC':
        compute_func = TTC_DRAC_MTTC
    else:
        print('Incorrect indicator. Please specify \'TTC\', \'DRAC\', \'MTTC\', or \'TTC_DRAC_MTTC\'.')
        return None
    import time as systime
    ts = []
    for _ in range(iterations):
        t = systime.time()
        _ = compute_func(samples, 'values')
        ts.append(systime.time()-t)
    return sum(ts)/iterations
