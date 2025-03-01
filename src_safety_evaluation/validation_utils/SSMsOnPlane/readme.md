# SSMsOnPlane

This folder contains scripts for calculating various Surrogate Safety Measures (SSMs) in two-dimensional space. All functions expect a pandas DataFrame called `samples` of vehicle pair samples, which should include the columns listed in the below table. The second input of all functions allows outputing 1) a dataframe with inputed samples plus new column(s) of the requested indicator, or 2) a numpy array of the requested indicator values.

| Variable : | Explanation                                                                                   |
|------------|-----------------------------------------------------------------------------------------------|
| x_i      : | x coordinate of the ego vehicle (usually assumed to be centroid)                              |
| y_i      : | y coordinate of the ego vehicle (usually assumed to be centroid)                              |
| vx_i     : | x coordinate of the velocity of the ego vehicle                                               |
| vy_i     : | y coordinate of the velocity of the ego vehicle                                               |
| hx_i     : | x coordinate of the heading direction of the ego vehicle                                      |
| hy_i     : | y coordinate of the heading direction of the ego vehicle                                      |
| acc_i    : | acceleration along the heading direction of the ego vehicle (only required if computing MTTC) |
| length_i : | length of the ego vehicle                                                                     |
| width_i  : | width of the ego vehicle                                                                      |
| x_j      : | x coordinate of another vehicle (usually assumed to be centroid)                              |
| y_j      : | y coordinate of another vehicle (usually assumed to be centroid)                              |
| vx_j     : | x coordinate of the velocity of another vehicle                                               |
| vy_j     : | y coordinate of the velocity of another vehicle                                               |
| hx_j     : | x coordinate of the heading direction of another vehicle                                      |
| hy_j     : | y coordinate of the heading direction of another vehicle                                      |
| acc_j    : | acceleration along the heading direction of another vehicle (optional)                        |
| length_j : | length of another vehicle                                                                     |
| width_j  : | width of another vehicle                                                                      |

## Geometry Utilities
[geometry_utils.py](src_safety_evaluation/validation_utils/SSMsOnPlane/geometry_utils.py) is a collection of functions for geometric calculations, where
- [`geometry_utils.intersect`](src_safety_evaluation/validation_utils/SSMsOnPlane/geometry_utils.py) finds the intersection of two lines;
- [`geometry_utils.getpoints`](src_safety_evaluation/validation_utils/SSMsOnPlane/geometry_utils.py) computes the four corners or front/rear points of each vehicle;
- [`geometry_utils.DTC_ij`](src_safety_evaluation/validation_utils/SSMsOnPlane/geometry_utils.py) calculates the euclidean distance to collision between two vehicles.  

## Longitudinal SSMs
[longitudinal_ssms.py](src_safety_evaluation/validation_utils/SSMsOnPlane/longitudinal_ssms.py) is a collection of functions for longitudinal SSMs adapted to be used in two-dimensional space, where
- [`longitudinal_ssms.TTC`](src_safety_evaluation/validation_utils/SSMsOnPlane/longitudinal_ssms.py) computes time-to-collision assuming constant speeds;
- [`longitudinal_ssms.DRAC`](src_safety_evaluation/validation_utils/SSMsOnPlane/longitudinal_ssms.py) computes deceleration rate to avoid collision;
- [`longitudinal_ssms.MTTC`](src_safety_evaluation/validation_utils/SSMsOnPlane/longitudinal_ssms.py) calculates modified time-to-collision considering accelerations.  
- [`longitudinal_ssms.PSD`](src_safety_evaluation/validation_utils/SSMsOnPlane/longitudinal_ssms.py) computes the proportion of stopping distance;
- [`longitudinal_ssms.TTC_DRAC_MTTC`](src_safety_evaluation/validation_utils/SSMsOnPlane/longitudinal_ssms.py) computes all three metrics together.  

## Two-Dimensional SSMs
[two_dimensional_ssms.py](src_safety_evaluation/validation_utils/SSMsOnPlane/two_dimensional_ssms.py) is a collection of functions for two-dimensional SSMs, where
- [`two_dimensional_ssms.TAdv`](src_safety_evaluation/validation_utils/SSMsOnPlane/two_dimensional_ssms.py) calculates the time advantage indicator;
- [`two_dimensional_ssms.get_ttc_components`](src_safety_evaluation/validation_utils/SSMsOnPlane/two_dimensional_ssms.py) obtains the longitudinal and lateral TTC components;
- [`two_dimensional_ssms.TTC2D`](src_safety_evaluation/validation_utils/SSMsOnPlane/two_dimensional_ssms.py) computes a 2D time-to-collision measure;
- [`two_dimensional_ssms.ACT`](src_safety_evaluation/validation_utils/SSMsOnPlane/two_dimensional_ssms.py) calculates the available collision time based on corners and edges of vehicles.  

## To use these functions 
Please import them and pass in a `samples` DataFrame with the appropriate columns. For example:

```python
import pandas as pd
from .longitudinal_ssms import TTC

df = pd.DataFrame(...)  # containing x_i, y_i, vx_i, etc.
results = TTC(df, toreturn='dataframe')
print(results[['TTC']])
```

## Notes
- The ego vehicle and another vehicle will never collide if they keep current speed when 
    1) indicator value is np.inf when the indicator is TTC, MTTC, TTC2D, TAdv, or ACT;
    2) indicator value is 0 when the indicator is DRAC.

- When indicator<0, the bounding boxes of the ego vehicle and another vehicle are overlapping.
This is due to approximating the space occupied by a vehicle with a rectangular.
In other words, negative indicator in this computation means the collision between the two 
vehicles almost (or although seldom, already) occurred.

- The computation can return extreme small positive values (for TTC/MTTC) or extreme large values (for DRAC) even when the vehivles overlap a bit (so should be negative values). In order to improve the accuracy, please use function CurrentD(samples, 'dataframe') or CurrentD(samples, 'values') to further exclude overlapping vehicles.



######################### Copyright (c) 2025 Yiru Jiao <y.jiao-1@tudelft.nl> ###########################
