'''
This script contains the functions to compute the distance matrix between time series data.
'''

import numpy as np
import torch
from tslearn.metrics import dtw, dtw_path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from joblib import Parallel, delayed


def get_EUC(X_tr, multivariate=True):
    X_tr[np.isnan(X_tr)] = 0
    if multivariate:
        X_tr = X_tr.reshape(len(X_tr), -1)
    if X_tr.shape[1] < 2000 and X_tr.shape[0] < 1000:
        return euclidean_distances(X_tr)
    else:
        X_tr = torch.tensor(X_tr).float() # use cpu is enough as transfering data between cpu and gpu can be time-consuming
        dist_mat = torch.cdist(X_tr, X_tr, p=2)
        return dist_mat.cpu().numpy()


def get_COS(X_tr, multivariate=True):
    X_tr[np.isnan(X_tr)] = 0
    if multivariate:
        return -cosine_similarity(X_tr.reshape(len(X_tr), -1))
    else:
        return -cosine_similarity(X_tr)


def parallel_dtw(i, X_tr, N):
    dist_row = np.zeros(N)
    for j in range(i+1, N):
        dist = dtw(X_tr[i], X_tr[j])
        dist_row[j] = dist
    return dist_row


def get_DTW(X_tr, multivariate=False):
    N = len(X_tr)
    dist_mat = np.zeros((N,N))
    if multivariate:
        desc = 'Get MDTW'
    else:
        desc = 'Get DTW'
    if N<100:
        for i in range(N):
            for j in range(i+1, N):  # Only iterate over the upper triangular matrix
                dist = dtw(X_tr[i], X_tr[j])
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
    else:
        progress_bar = tqdm(range(N), desc=desc, ascii=True, miniters=int(N/10))
        dist_rows = Parallel(n_jobs=-1)(delayed(parallel_dtw)(i, X_tr, N) for i in progress_bar)
        dist_mat_upper = np.array(dist_rows)
        dist_mat = dist_mat_upper + dist_mat_upper.T
        
    return dist_mat


def save_sim_mat(X_tr, multivariate=False, dist_metric='DTW', min_ = 0, max_ = 1):
    if dist_metric=='DTW':
        dist_mat = get_DTW(X_tr, multivariate)
    elif dist_metric=='COS':
        dist_mat = get_COS(X_tr, multivariate)
    elif dist_metric=='EUC':
        dist_mat = get_EUC(X_tr, multivariate)
        
    # fill diagonal with the minimum value of the off-diagonal elements
    try:
        np.fill_diagonal(dist_mat, dist_mat[np.nonzero(dist_mat)].min())
        # normalize the distance
        dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
    except: # if all elements are zero
        dist_mat = dist_mat
    
    # return similarity matrix, ensure positive values
    return abs(1 - dist_mat)
