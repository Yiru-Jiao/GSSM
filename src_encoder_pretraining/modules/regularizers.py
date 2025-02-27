'''
This script contains the implementation of the topological and geometric regularisers.
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.modules.loss_utils import *


def topo_loss(model, x):
    # encode using model
    latent = model.encode(x)

    # compute and normalize distances in the original sapce and latent space
    x_distances = topo_euclidean_distance_matrix(x) # (B, N, N)
    x_distances = x_distances / x_distances.max()
    latent_distances = topo_euclidean_distance_matrix(latent) # (B, N, N)
    latent_distances = latent_distances / latent_distances.max()

    # compute topological signature distance
    topo_sig = TopologicalSignatureDistance()
    topo_error = topo_sig(x_distances, latent_distances)

    # normalize topo_error according to batch_size
    batch_size = x.size()[0]
    topo_error = topo_error / float(batch_size)

    return topo_error


def geo_loss(model, x, bandwidth):
    # encode using model
    latent = model.encode(x)
    if latent.size(1) == 5: # need to reshape into (B, 25(24->0), 16) to map with the original time series
        latent = latent[:,::-1,:].reshape(latent.size(0), 25, -1)

    if len(x.size()) == 2: # (B, N) -> (B, N, 1) for encoding current features
        x = x.unsqueeze(-1)
    L = get_laplacian(x, bandwidth=bandwidth)
    H_tilde = get_JGinvJT(L, latent)
    iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde)

    return iso_loss
