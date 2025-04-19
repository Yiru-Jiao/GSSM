'''
This file contains functions to calculate evaluation metrics for safety warnings.
'''

import numpy as np
import pandas as pd


def get_statistics(warning, return_statistics=False):
    # true_positives = warning[warning['danger_recorded']&(warning['true_warning']>0.5)].groupby('threshold').size()
    # false_positives = warning[warning['safety_recorded']&(warning['false_warning']>0.5)].groupby('threshold').size()
    # true_negatives = warning[warning['safety_recorded']&(warning['false_warning']<0.5)].groupby('threshold').size()
    # false_negatives = warning[warning['danger_recorded']&(warning['true_warning']<0.5)].groupby('threshold').size()
    true_positives = warning[warning['danger_recorded']&(warning['true_warning']>0.5)].groupby('threshold').size()
    false_positives = warning[warning['safety_recorded']].groupby('threshold')['num_false_warning'].sum()
    true_negatives = warning[warning['safety_recorded']].groupby('threshold')['num_true_non_warning'].sum()
    false_negatives = warning[warning['danger_recorded']&(warning['true_warning']<0.5)].groupby('threshold').size()
    statistics = pd.concat([true_positives, false_positives, true_negatives, false_negatives], axis=1, keys=['TP', 'FP', 'TN', 'FN'])
    statistics = statistics.fillna(0) # nan can be caused by empty combination of threshold and warning
    if return_statistics:
        return statistics.reset_index().sort_values('TP')
    else:
        statistics = statistics.sort_index()
        return statistics['TP'], statistics['FP'], statistics['TN'], statistics['FN']
    

def area_curve(xaxis, yaxis, threshold):
    sorted_indices = np.argsort(xaxis)
    xaxis = xaxis[sorted_indices]
    yaxis = yaxis[sorted_indices]
    virtual_xaxis = np.linspace(0, 1, 1000)
    virtual_yaxis = np.interp(virtual_xaxis, xaxis, yaxis)
    virtual_yaxis = virtual_yaxis[virtual_xaxis<=threshold]
    virtual_xaxis = virtual_xaxis[virtual_xaxis<=threshold]
    area = np.trapz(virtual_yaxis, virtual_xaxis)
    return area


def get_time(warning, threshold=1.5):
    warning = warning.copy()
    warning['TTI'] = warning['impact_time'] - warning['warning_timestamp']/1000
    median_TTI = warning.groupby('threshold')['TTI'].median()
    PTTI = len(warning[warning['TTI']>=threshold])/len(warning)
    return median_TTI, PTTI


def get_eval_metrics(warning, thresholds={'roc':[0.1,0.2],'tti':None}):
    # Get confusion matrix statistics
    tp, fp, tn, fn = get_statistics(warning, return_statistics=False)
    # Based on Precision-Recall curve
    auprc = area_curve((tp/np.maximum(1e-6, tp+fn)).values,
                       (tp/np.maximum(1e-6, tp+fp)).values, 1)
    # Based on Receiver Operating Characteristic curve
    roc_metrics = dict()
    for threshold in thresholds['roc']:
        aroc = area_curve((fp/np.maximum(1e-6, fp+tn)).values, 
                          (tp/np.maximum(1e-6, tp+fn)).values, threshold) / threshold
        roc_metrics[f'aroc_{int(threshold*100)}'] = aroc
    # Based on F1-mTTI curve
    if thresholds['tti'] is not None:
        f1 = tp / (tp + 0.5*(fp + fn))
        mTTI, PTTI = get_time(warning, thresholds['tti'])
        f1_mTTI = pd.concat([f1, mTTI], axis=1, keys=['f1', 'mTTI'])
        mTTI_star = f1_mTTI.loc[f1_mTTI['f1'].idxmax()]['mTTI']
    else:
        mTTI_star = None
        PTTI = None

    # Create dataframe to store metrics
    metrics = {**roc_metrics, **{'auprc': auprc, 'mTTI_star': mTTI_star, 'PTTI': PTTI}}
    return metrics
