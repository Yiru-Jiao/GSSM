'''
This file contains functions to calculate evaluation metrics for safety warnings.
'''

import numpy as np
import pandas as pd
small_eps = 1e-6


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


def partial_auc(fpr, tpr, min_tpr=0.80, resolution=1000, normalize=True):
    '''
    Integrate the ROC curve only where TPR >= min_tpr
    '''
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    # sort the points by FPR
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]

    # interpolate onto a dense, uniform FPR grid
    virtual_fpr = np.linspace(small_eps, 1 - small_eps, resolution)
    interp_tpr = np.interp(virtual_fpr, fpr, tpr)

    # keep only the part above the TPR floor
    mask = interp_tpr >= min_tpr
    if not mask.any():
        return 0.0
    
    area = np.trapz(interp_tpr[mask] - min_tpr, virtual_fpr[mask])
    if normalize:
        area /= (1.0 - min_tpr)            # theoretical max area
    return area


def get_auprc(recall, precision, resolution=1000):
    '''
    Area under the Precision-Recall curve
    '''
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)

    order = np.argsort(recall)
    recall, precision = recall[order], precision[order]

    virtual_recall = np.linspace(small_eps, 1 - small_eps, resolution)
    interp_precision  = np.interp(virtual_recall, recall, precision)

    return np.trapz(interp_precision, virtual_recall)


def partial_prc(recall, precision, min_recall=0.80):
    '''
    Get the maximum Precision when Recall >= min_recall
    '''
    recall    = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)

    # keep only the part above the Recall floor
    mask = recall >= min_recall
    if not mask.any():
        return 0.0
    return precision[mask].max()


def get_time(warning, f1=None, cutoff=1.5):
    '''
    Median time-to-impact (TTI) and the proportion with TTI ≥ cutoff.
    '''
    w = warning.copy()
    w['TTI'] = w['impact_time'] - w['warning_timestamp'] / 1000.0

    if f1 is None:
        tp, fp, tn, fn = get_statistics(warning, return_statistics=False)
        f1 = tp / np.maximum(small_eps, tp + 0.5*(fp + fn))

    median_tti = w.groupby('threshold')['TTI'].median()
    mtti_star = median_tti.loc[f1.idxmax()]
    w = w[w['threshold'] == f1.idxmax()]
    ptti_star = len(w[w['TTI']>=cutoff])/len(w[~w['TTI'].isna()])
    return mtti_star, ptti_star


def get_eval_metrics(warning, thresholds={'roc': [0.80, 0.90], 'prc':[0.80, 0.90], 'tti': None}):
    '''
    Compute safety-oriented evaluation metrics from a dataframe with
    per-threshold confusion counts stored in columns tp, fp, tn, fn.
    '''
    # confusion-matrix stats
    tp, fp, tn, fn = get_statistics(warning, return_statistics=False)

    # ROC
    roc_metrics = {}
    fpr = fp / np.maximum(small_eps, fp + tn)
    tpr = tp / np.maximum(small_eps, tp + fn)
    for threshold in thresholds['roc']:
        roc_metrics[f'aroc_{int(threshold*100)}'] = partial_auc(fpr, tpr, threshold)

    # PRC
    prc_metrics = {}
    recall = tp / np.maximum(small_eps, tp + fn)
    precision = tp / np.maximum(small_eps, tp + fp)
    auprc = get_auprc(recall, precision)
    for threshold in thresholds['prc']:
        prc_metrics[f'aprc_{int(threshold*100)}'] = partial_prc(recall, precision, threshold)

    # ATC
    if thresholds['tti'] is not None:
        f1 = tp / np.maximum(small_eps, tp + 0.5*(fp + fn))
        mtti_star, ptti_star = get_time(warning, f1, thresholds['tti'])
    else:
        mtti_star = None
        ptti_star = None

    return {
        **roc_metrics,
        **prc_metrics,
        'auprc': auprc,
        'PTTI_star': ptti_star,
        'mTTI_star': mtti_star,
    }