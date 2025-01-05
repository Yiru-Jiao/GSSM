'''
This file contains the implementation of the topological and geometric regularizers.
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.modules.loss_utils import *


def topo_loss(model, x):
    # encode using model
    latent = model.encode(x, **model.encode_args)

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

    L = get_laplacian(x, bandwidth=bandwidth)
    H_tilde = get_JGinvJT(L, latent)
    iso_loss = relaxed_distortion_measure_JGinvJT(H_tilde)

    return iso_loss
