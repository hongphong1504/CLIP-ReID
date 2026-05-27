from abc import ABC, abstractmethod

import torch
import numpy as np

from utils.distance import euclidean_distance
from utils.reranking import re_ranking
from utils.feature_ops import normalize
from .refiner import UFFMRefiner
from .rank import eval_func


class BaseEvaluator(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, feats, pids, camids):
        pass

    @abstractmethod
    def evaluate(self):
        pass




class StandardEvaluator(BaseEvaluator):

    def __init__(
        self,
        num_query,
        max_rank=50,
        feat_norm='yes'
    ):

        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, feats, pids, camids):
        self.feats.append(feats.cpu())
        self.pids.extend(np.asarray(pids))
        self.camids.extend(np.asarray(camids))

    def _build_features(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test features are normalized")
            feats = normalize(feats)

        qf = feats[:self.num_query]
        gf = feats[self.num_query:]

        q_pids = np.asarray(self.pids[:self.num_query])

        g_pids = np.asarray(self.pids[self.num_query:])

        q_camids = np.asarray(self.camids[:self.num_query])

        g_camids = np.asarray(self.camids[self.num_query:])

        return (
            qf,
            gf,
            q_pids,
            g_pids,
            q_camids,
            g_camids
        )

    def compute_distance(self, qf, gf):
        return euclidean_distance(qf, gf)

    def evaluate(self):
        (qf, gf, q_pids, g_pids, q_camids, g_camids) = self._build_features()

        distmat = self.compute_distance(qf, gf)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank)

        return cmc, mAP
    

class AdvancedEvaluator(StandardEvaluator):

    def __init__(self, cfg, num_query, alpha=None, beta=None, theta=None, max_rank=50, feat_norm='yes'):
        super().__init__(num_query, max_rank, feat_norm)
        self.cfg = cfg
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def compute_distance(self, qf, gf):
        method = self.cfg.TEST.METHOD.lower()

        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_camids = np.asarray(self.camids[self.num_query:])

        if method == "rerank":
            return re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        if method == "uffm":
            refiner = UFFMRefiner(self.cfg)

            return refiner.refine(
                qf=qf,
                gf=gf,
                q_camids=q_camids,
                g_camids=g_camids,
                q_ids=q_pids
            )

        if method == "uffm_amc":
            refiner = UFFMRefiner(self.cfg)

            return refiner.refine(
                qf=qf,
                gf=gf,
                q_camids=q_camids,
                g_camids=g_camids,
                q_ids=q_pids,
                alpha=self.alpha,
                beta=self.beta,
                theta=self.theta
            )