'''
This script defines the loss functions for contrastive learning for profiles.
The hierarchical contrastive loss reuses and adapts the code of TS2Vec and SoftCLT.
TS2Vec https://github.com/zhihanyue/ts2vec
SoftCLT https://github.com/seunghan96/softclt
'''

import os
import sys
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_encoder_pretraining.modules.regularizers import *


def combined_loss(model, x, loss_config, regularizer_config):
    temporal_unit = loss_config['temporal_unit']
    out1, out2 = mask_and_crop(model._net, x, temporal_unit)
    loss_scl = hierarchical_contrastive_loss(out1, out2, **loss_config)
    loss_components = []

    if regularizer_config['reserve'] is None:
        loss = loss_scl
        loss_components.append(loss_scl)
    else:
        loss = 0.5 * torch.exp(-model.loss_log_vars[0]) * loss_scl*(1-torch.exp(-loss_scl)) + 0.5 * model.loss_log_vars[0]
        loss_components.append(loss_scl)
        loss_components.append(model.loss_log_vars[0])
        
        if regularizer_config['reserve'] == 'topology':
            loss_topo_regularizer = topo_loss(model, x)
            loss += 0.5 * torch.exp(-model.loss_log_vars[1]) * loss_topo_regularizer*(1-torch.exp(-loss_topo_regularizer)) + 0.5 * model.loss_log_vars[1]
            loss_components.append(loss_topo_regularizer)
            loss_components.append(model.loss_log_vars[1])
        elif regularizer_config['reserve'] == 'geometry':
            loss_geo_regularizer = geo_loss(model, x, regularizer_config['bandwidth'])
            loss += 0.5 * torch.exp(-model.loss_log_vars[1]) * loss_geo_regularizer*(1-torch.exp(-loss_geo_regularizer)) + 0.5 * model.loss_log_vars[1]
            loss_components.append(loss_geo_regularizer)
            loss_components.append(model.loss_log_vars[1])
        else:
            raise ValueError('Undefined regularizer, should be either "topology", or "geometry"')    
    return loss, loss_components


def instance_contrastive_loss(z1, z2, soft_or_hard=('hard',)):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    if soft_or_hard[0] == 'hard':
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    elif soft_or_hard[0] == 'soft':
        soft_labels_L, soft_labels_R = soft_or_hard[1], soft_or_hard[2]
        loss = torch.sum(logits[:,i]*soft_labels_L)
        loss += torch.sum(logits[:,B + i]*soft_labels_R)
        loss /= (2*B*T)

    return loss


def temporal_contrastive_loss(z1, z2, soft_or_hard=('hard',)):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    if soft_or_hard[0] == 'hard':
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    elif soft_or_hard[0] == 'soft':
        timelag_L, timelag_R = soft_or_hard[1], soft_or_hard[2]
        loss = torch.sum(logits[:,t]*timelag_L)
        loss += torch.sum(logits[:,T + t]*timelag_R)
        loss /= (2*B*T)
        
    return loss


def hierarchical_contrastive_loss(z1, z2, temporal_unit=0, lambda_inst=0.5, 
                                  soft_labels=None, tau_inst=0, tau_temp=0, temporal_hierarchy=None,
                                  ):
    
    if soft_labels is not None:
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)

    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_inst != 0:
            if tau_inst > 0:
                soft_or_hard = ('soft', soft_labels_L, soft_labels_R)
            else:
                soft_or_hard = ('hard',)
            loss += lambda_inst * instance_contrastive_loss(z1, z2, soft_or_hard)

        if d >= temporal_unit and 1 - lambda_inst != 0:
            if tau_temp > 0:
                if temporal_hierarchy is None:
                    timelag = timelag_sigmoid(z1, tau_temp)
                else:
                    if temporal_hierarchy=='exponential':
                        timelag = timelag_sigmoid(z1, tau_temp*(2**d)) # 2**d because kernel_size in max_pool1d is 2
                    elif temporal_hierarchy=='linear':
                        timelag = timelag_sigmoid(z1, tau_temp*(d+1))
                    else:
                        raise ValueError('Undefined temporal_hierarchy, should be either "exponential" or "linear"')
                    
                timelag_L, timelag_R = dup_matrix(timelag)
                soft_or_hard = ('soft', timelag_L, timelag_R)
            else:
                soft_or_hard = ('hard',)
            loss += (1 - lambda_inst) * temporal_contrastive_loss(z1, z2, soft_or_hard)

        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1 and lambda_inst != 0:
        if tau_inst > 0:
            loss += lambda_inst * instance_contrastive_loss(z1, z2, ('soft', soft_labels_L, soft_labels_R))
        else:
            loss += lambda_inst * instance_contrastive_loss(z1, z2)
        d += 1

    return loss / d

