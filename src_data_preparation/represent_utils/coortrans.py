'''
This file contains the methods for coordinate transformation for interaction extraction.
'''

import numpy as np
import pandas as pd


class coortrans():
    def __init__(self):
        pass


    def rotate_coor(self, xyaxis, yyaxis, x2t, y2t):
        '''
        Rotate the coordinates (x2t, y2t) to the coordinate system with the y-axis along (xyaxis, yyaxis).

        Parameters:
        - xyaxis: x-coordinate of the y-axis in the new coordinate system
        - yyaxis: y-coordinate of the y-axis in the new coordinate system
        - x2t: x-coordinate to be rotated
        - y2t: y-coordinate to be rotated

        Returns:
        - x: rotated x-coordinate
        - y: rotated y-coordinate
        '''
        x = yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t-xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
        y = xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t+yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
        return x, y


    def transform_coor(self, pairs, view, surrounding=None):
        '''
        Transform the coordinates of the pairs to the relative view or the view of a selected ego vehicle.

        Parameters:
        - pairs: DataFrame containing the pairs of coordinates
        - view: 'relative' for relative view or 'ego' for view of ego vehicle or 'sur' for view of surrounding vehicle

        Returns:
        - transformed pairs DataFrame
        '''
        if 'timestamp' in pairs.columns:
            time_ref = 'timestamp'
        elif 'time' in pairs.columns:
            time_ref = 'time'
        else:
            time_ref = None
        
        # Swap the label of 'ego' and 'sur' if the view is 'sur'
        if view == 'sur':
            dict_rename = {key: key.replace('_ego','_intermediate').replace('_sur','_ego').replace('_intermediate','_sur') for key in pairs.columns}
            if time_ref is not None:
                dict_rename.pop(time_ref, None)
            pairs = pairs.rename(columns=dict_rename)

        # Determine transformation reference based on the view
        if view == 'relative':
            coor_ref = pd.DataFrame({'x_axis': pairs['v_ego']*pairs['hx_ego']-pairs['v_sur']*pairs['hx_sur'],
                                     'y_axis': pairs['v_ego']*pairs['hy_ego']-pairs['v_sur']*pairs['hy_sur'], 
                                     'x_origin': pairs['x_ego'], 
                                     'y_origin': pairs['y_ego']}, index=pairs.index)
            condition = (coor_ref['x_axis']==0)&(coor_ref['y_axis']==0)
            coor_ref.loc[condition, ['x_axis','y_axis']] = pairs.loc[condition, ['hx_ego','hy_ego']].values
        else:
            # Calculate the reference coordinate system based on the view of vehicle i
            coor_ref = pd.DataFrame({'x_axis': pairs['hx_ego'], 
                                     'y_axis': pairs['hy_ego'],
                                     'x_origin': pairs['x_ego'],
                                     'y_origin': pairs['y_ego']}, index=pairs.index)
        
        # Rotate the coordinates of pairs and update the DataFrame
        x_ego, y_ego = np.zeros(len(pairs)), np.zeros(len(pairs))
        x_sur, y_sur = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], pairs['x_sur']-coor_ref['x_origin'], pairs['y_sur']-coor_ref['y_origin'])
        pairs = pairs.assign(x_ego=x_ego, y_ego=y_ego, x_sur=x_sur, y_sur=y_sur)
        for obj in ['ego', 'sur']:
            x, y = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], pairs[f'hx_{obj}'], pairs[f'hy_{obj}'])
            pairs[f'hx_{obj}'] = x
            pairs[f'hy_{obj}'] = y

        if surrounding is None:
            return pairs
        else:
            # Rotate the coordinates of the surroundings and update the DataFrame
            coor_ref['y_sur'] = pairs['y_sur']
            coor_ref = coor_ref.loc[surrounding.index.values].reset_index()
            surrounding = surrounding.reset_index()
            x, y = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], surrounding['x']-coor_ref['x_origin'], surrounding['y']-coor_ref['y_origin'])
            hx, hy = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], surrounding['hx'], surrounding['hy'])
            surrounding = surrounding.assign(x=x, y=y, hx=hx, hy=hy)
            if 'timestamp' in surrounding.columns:
                time_ref = 'timestamp'
            elif 'time' in surrounding.columns:
                time_ref = 'time'
            return pairs, surrounding.set_index(time_ref)
    

    def angle(self, vec1x, vec1y, vec2x, vec2y):
        '''
        Calculate the angle between two vectors.

        Parameters:
        - vec1x: x-component of the first vector
        - vec1y: y-component of the first vector
        - vec2x: x-component of the second vector
        - vec2y: y-component of the second vector

        Returns:
        - angle: angle between the two vectors
        '''
        sin = vec1x * vec2y - vec2x * vec1y
        cos = vec1x * vec2x + vec1y * vec2y
        return np.arctan2(sin, cos)
        
