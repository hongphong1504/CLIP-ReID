import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) \
               + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)

    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf, eps=1e-5):
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + eps, 1 - eps)
    dist_mat = np.arccos(dist_mat)
    return dist_mat





