'''
This module contains utility functions and classes for computing the contrastive loss and regularizers.
Functions and classes are adjusted from the original implementation in the SoftCLT repository, TopoAE repository, and GGAE repository.
SoftCLT https://github.com/seunghan96/softclt
TopoAE https://github.com/BorgwardtLab/topological-autoencoders
GGAE https://github.com/JungbinLim/GGAE-public
'''

import torch
import random


#####################################
## functions for contrastive_loss  ##
#####################################

def take_per_row(A, indx, num_elem):
    """
    Selects a specified number of elements per row from a 2D tensor.
    """
    all_indx = indx[:, None] + torch.arange(num_elem, device=A.device)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def dup_matrix(mat):
    """
    Duplicates a matrix by appending its transpose to itself.
    """
    mat0 = torch.tril(mat, diagonal=-1)[:, :-1]   
    mat0 += torch.triu(mat, diagonal=1)[:, 1:]
    mat1 = torch.cat([mat0, mat], dim=1)
    mat2 = torch.cat([mat, mat0], dim=1)
    return mat1, mat2


def mask_and_crop(encoder, x, temporal_unit=0):
    """
    Crops the masked input tensor and computes the output of the encoder.
    Note: input mask is embedded in the encoder.
    """
    ts_l = x.size(1)
    if encoder.training:
        crop_l = random.randint(2**(temporal_unit+1), ts_l)
        crop_left = random.randint(0, ts_l-crop_l)
        crop_right = crop_left + crop_l
        crop_eleft = random.randint(0, crop_left)
        crop_eright = random.randint(crop_right, ts_l)
        crop_offset = torch.randint(-crop_eleft, ts_l-crop_eright+1, size=(x.size(0),), device=x.device)

        out1 = encoder(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
        out1 = out1[:, -crop_l:]
        
        out2 = encoder(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
        out2 = out2[:, :crop_l]
    else:
        out1 = encoder(x)
        out2 = out1

    return out1, out2


def timelag_sigmoid(z1, sigma=1):
    """
    Computes a time-lagged sigmoid matrix based on the input tensor.
    """
    T = z1.size(1)
    dist = torch.arange(T, device=z1.device).float()
    dist = torch.abs(dist[:, None] - dist[None, :])
    matrix = 2 / (1 + torch.exp(dist*sigma))
    matrix = torch.where(matrix < 1e-6, torch.zeros_like(matrix), matrix)  # set very small values to 0
    return matrix


###############################################################
## functions and classes for topology preserving regularizer ##
###############################################################

def topo_euclidean_distance_matrix(x, p=2):
    """
    Computes the pairwise Euclidean distance matrix between the rows of a 2D tensor.
    """
    x_flat = x.view(x.size(0), -1)
    x_flat[torch.isnan(x_flat)] = 0
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    return distances


class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''
        self._parent = torch.arange(n_vertices, dtype=torch.long)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''
        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''
        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''
        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __call__(self, matrix):
        """
        Computes the persistence pairs of the given distance matrix.
        """
        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = torch.triu_indices(n_vertices, n_vertices)
        edge_weights = matrix[triu_indices[0], triu_indices[1]]
        edge_indices = torch.argsort(edge_weights, stable=True)

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        return torch.tensor(persistence_pairs)


class TopologicalSignatureDistance(torch.nn.Module):
    def __init__(self, match_edges='symmetric'):
        """
        Topological signature computation.
        """
        super().__init__()
        self.match_edges = match_edges

        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        """
        Returns pairings for 0-dimensional features (ignore cycles).
        """
        return self.signature_calculator(distances)

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        """
        Selects distances from the distance matrix based on the given pairs.
        """
        selected_distances = distance_matrix[(pairs[:, 0], pairs[:, 1])]

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """
        Compute distance between two topological signatures.
        """
        return ((signature1 - signature2)**2).sum(dim=-1)

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """
        Return topological distance of two pairwise distance matrices.
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance = distance1_2 + distance2_1

        return distance


###############################################################
## functions and classes for geometry preserving regularizer ##
###############################################################

def get_laplacian(X, bandwidth=50): # bandwidth tuning should increase exponentially like bw**2
    """
    Calculate the Normalized Graph Laplacian for a given set of data points.
    """
    B, N, _ = X.shape
    c = 1/4

    X[torch.isnan(X)] = 0
    dist_XX = torch.cdist(X, X, p=2)
    K = torch.exp(-dist_XX**2 / bandwidth)
    d_i = K.sum(dim=1)
    D_inv = torch.diag_embed(1/d_i)
    K_tilde = D_inv @ K @ D_inv
    d_i_tilde = K_tilde.sum(dim=1)
    D_tilde_inv = torch.diag_embed(1/d_i_tilde)
    I = torch.diag_embed(torch.ones(B, N, device=X.device))
    L = (D_tilde_inv@K_tilde - I)/(c*bandwidth)

    return L # (B, N, N) or (B, T, T)


def get_JGinvJT(L, Y):
    """
    Calculate the JGinvJT matrix for each data point.
    """    
    Batch, N, n = Y.shape

    if N*n*n <= 1e6:
        L_mul_Y = L @ Y

        Y_expanded = Y.unsqueeze(-1)
        YT_expanded = Y.unsqueeze(-2)

        term1 = (L @ (Y_expanded * YT_expanded).view(Batch, N, n * n)).view(Batch, N, n, n)
        term2 = Y_expanded * L_mul_Y.unsqueeze(-2)
        term3 = YT_expanded * L_mul_Y.unsqueeze(-1)

        H_tilde = 0.5 * (term1 - term2 - term3)
    else:
        H_tilde_list = []
        for i in range(Batch):
            L_sub = L[i]
            Y_sub = Y[i]

            L_mul_Y = L_sub @ Y_sub

            Y_expanded = Y_sub.unsqueeze(-1)
            YT_expanded = Y_sub.unsqueeze(-2)

            term1 = (L_sub @ (Y_expanded * YT_expanded).view(N, n * n)).view(N, n, n)
            term2 = Y_expanded * L_mul_Y.unsqueeze(-2)
            term3 = YT_expanded * L_mul_Y.unsqueeze(-1)

            H_tilde_sub = 0.5 * (term1 - term2 - term3)
            H_tilde_list.append(H_tilde_sub)

        H_tilde = torch.stack(H_tilde_list, dim=0)

    return H_tilde


def relaxed_distortion_measure_JGinvJT(H):
    """
    Calculate the relaxed distortion measure for a given JGinvJT matrix.
    """
    TrH = H.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    TrH2 = (H @ H).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    distortion = (TrH2).mean() - 2 * (TrH).mean()
    return distortion

