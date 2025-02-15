To compute two-dimensional Time-To-Collision,
use function TTC(samples, 'dataframe') or TTC(samples, 'values');

To compute two-dimensional Deceleration Rate to Avoid Collision,
use function DRAC(samples, 'dataframe') or DRAC(samples, 'values');

To compute two-dimensional Modified Time-To-Collision,
use function MTTC(samples, 'dataframe') or MTTC(samples, 'values');

To compute all the above three indicators at once,
use function TTC_DRAC_MTTC(samples, 'dataframe') or TTC_DRAC_MTTC(samples, 'values').

The first input `samples` is a pandas dataframe of vehicle pair samples, 
which should include the following columns:
------------------------------------------------------------------------------------------------------------
x_i      :  x coordinate of the ego vehicle (usually assumed to be centroid)                               |
y_i      :  y coordinate of the ego vehicle (usually assumed to be centroid)                               |
vx_i     :  x coordinate of the velocity of the ego vehicle                                                |
vy_i     :  y coordinate of the velocity of the ego vehicle                                                |
hx_i     :  x coordinate of the heading direction of the ego vehicle                                       |
hy_i     :  y coordinate of the heading direction of the ego vehicle                                       |
acc_i    :  acceleration along the heading direction of the ego vehicle (only required if computing MTTC)  |
length_i :  length of the ego vehicle                                                                      |
width_i  :  width of the ego vehicle                                                                       |
x_j      :  x coordinate of another vehicle (usually assumed to be centroid)                               |
y_j      :  y coordinate of another vehicle (usually assumed to be centroid)                               |
vx_j     :  x coordinate of the velocity of another vehicle                                                |
vy_j     :  y coordinate of the velocity of another vehicle                                                |
hx_j     :  x coordinate of the heading direction of another vehicle                                       |
hy_j     :  y coordinate of the heading direction of another vehicle                                       |
acc_j    :  acceleration along the heading direction of another vehicle (optional)                         |
length_j :  length of another vehicle                                                                      |
width_j  :  width of another vehicle                                                                       |
------------------------------------------------------------------------------------------------------------
The second input allows outputing 
    1) a dataframe with inputed samples plus new column(s) of the requested indicator, or
    2) a numpy array of the requested indicator values.

The ego vehicle and another vehicle will never collide if they keep current speed when 
    1) indicator==np.inf when indicator==TTC or indicator==MTTC, or
    2) indicator==0 when indicator==DRAC.

When indicator<0, the bounding boxes of the ego vehicle and another vehicle are overlapping.
This is due to approximating the space occupied by a vehicle with a rectangular.
In other words, negative indicator in this computation means the collision between the two 
vehicles almost (or although seldom, already) occurred.

*** Note that the computation can return extreme small positive values (for TTC/MTTC) or 
    extreme large values (for DRAC) even when the vehivles overlap a bit (so should be negative values). 
    In order to improve the accuracy, please use function CurrentD(samples, 'dataframe') or 
    CurrentD(samples, 'values') to further exclude overlapping vehicles.

######################### Copyright (c) 2025 Yiru Jiao <y.jiao-1@tudelft.nl> ###########################
