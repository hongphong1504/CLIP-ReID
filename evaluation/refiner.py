from abc import ABC, abstractmethod
import torch
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from utils.feature_ops import normalize

class BaseRefiner(ABC):

    @abstractmethod
    def refine(self, qf, gf, q_camids, g_camids):
        pass


class AMCEstimator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE

    @torch.no_grad()
    def extract_features(
        self,
        model,
        train_loader,
        caption_dict
    ):

        model.eval()

        feats = []
        pids = []
        camids = []

        for batch in tqdm(train_loader, desc="ExtractTrainFeat"):

            img, pid, camid, view_id, img_path = batch

            img = img.to(self.device)
            captions = [caption_dict[p] for p in img_path]

            if self.cfg.MODEL.SIE_CAMERA:
                camid = camid.to(self.device)
            else:
                camid = None

            if self.cfg.MODEL.SIE_VIEW:
                view_id = view_id.to(self.device)
            else:
                view_id = None

            feat = model(image=img, caption=captions, cam_label=camid, view_label=view_id)

            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(camid.cpu().numpy())

        feats = torch.cat(feats, dim=0)

        if self.cfg.TEST.FEAT_NORM == 'yes':
            feats = normalize(feats)

        return (
            feats,
            np.asarray(pids),
            np.asarray(camids)
        )
    
    def fit(self, model, train_loader, caption_dict, n_data=1000, rand_seed=1234):
        print("Fitting AMC (n={})...".format(n_data))
        feats, f_ids, f_camids = self.extract_features(model, train_loader, caption_dict)

        np.random.seed(rand_seed)

        positives = []
        negatives = []

        pid = np.unique(f_ids)

        valid_pos_pids = [p for p in pid if len(np.where(f_ids == p)[0]) >= 3]
        if len(valid_pos_pids) == 0:
            raise ValueError(
                "No IDs have at least 3 images required for Positive Center calculation." 
            )

        for ii in tqdm(range(n_data), desc="PositiveData: "):
            m = np.random.choice(valid_pos_pids, 1)[0]
            index_arr = np.where(f_ids == m)[0]

            chosen_idx = np.random.choice(
                index_arr, size=len(index_arr), replace=False
            )

            point1 = feats[chosen_idx[0]].view(1, -1)
            point2 = feats[chosen_idx[1]].view(1, -1)
            centers = torch.mean(feats[chosen_idx[2:]], dim=0).view(1, -1)

            cce = 1.0 if f_camids[chosen_idx[0]] == f_camids[chosen_idx[1]] else 0.0

            cos_sim = float(torch.matmul(point1, point2.t()))
            cos_sim_c = float(torch.matmul(point1, centers.t()))
            positives.append([cos_sim, cos_sim_c, cce])

        for ii in tqdm(range(n_data), desc="NegativeData: "):
            m, n = np.random.choice(pid, 2, replace=False)

            index_arr_m = np.where(f_ids == m)[0]
            index_arr_n = np.where(f_ids == n)[0]

            idx_m = np.random.choice(index_arr_m, 1)[0]
            idx_n = np.random.choice(index_arr_n, size=len(index_arr_n), replace=False)

            point1 = feats[idx_m].view(1, -1)
            point2 = feats[idx_n[0]].view(1, -1)
            centers = torch.mean(feats[idx_n[1:]], dim=0).view(1, -1)

            cce = 1.0 if f_camids[idx_m] == f_camids[idx_n[0]] else 0.0

            cos_sim = float(torch.matmul(point1, point2.t()))
            cos_sim_c = float(torch.matmul(point1, centers.t()))
            negatives.append([cos_sim, cos_sim_c, cce])

        X = np.concatenate((positives, negatives), axis=0)
        Y = np.concatenate(
            (np.ones(len(positives)), np.full(len(negatives), -1)), axis=0)

        reg = LinearRegression()
        model_reg = reg.fit(X, Y)

        score = model_reg.score(X, Y)
        score_pos = model_reg.score(positives, np.ones(len(positives)))
        score_neg = model_reg.score(negatives, np.full(len(negatives), -1))

        alpha, beta, theta = model_reg.coef_

        return alpha, beta, theta
    

class UFFMRefiner(BaseRefiner):

    def __init__(self, cfg):
        self.cfg = cfg

    def refine(
        self,
        qf,
        gf,
        q_camids,
        g_camids,
        q_ids,
        alpha=None,
        beta=None,
        theta=None
    ):
        k = self.cfg.TEST.UFFM_K
        print(f"Calculating similarity with UFFM (k={k})...")

        camid2idx = defaultdict(list)
        index_g = [] 
        for idx, camid in enumerate(g_camids):
            index_g.append(idx)
            camid2idx[camid].append(idx)

        # Get unique camera IDs for the queries and their count
        unique_camids_q = sorted(np.unique(q_camids))
        num_camid_q = len(unique_camids_q)

        # Initialize tensors to store similarity matrices and a dictionary for uncertain multi-view features
        dict_umvf = {}
        cce = {}

        # Compute similarity between all gallery features
        sim_gg = torch.mm(gf, gf.t())

        # Loop through each unique camera ID in the query
        for indx in unique_camids_q:
            # Find indices of gallery features with the same camera ID as the current query
            ind_qcamid = camid2idx[indx] 
            
            # Exclude gallery features with the same camera ID as the query
            ind_exq = np.setdiff1d(index_g, ind_qcamid)  
            gf_new = gf[ind_exq]  
            sim_gg_exq = sim_gg[:, ind_exq]  
            sim_gg_argsort = torch.argsort(1 - sim_gg_exq, dim=1)
            sim_gg_argtopk = sim_gg_argsort[:, :k]  

            # Calculate weights for the top-k gallery features
            sim_gg_topk = sim_gg_exq[torch.arange(sim_gg_exq.size(0)).unsqueeze(1), sim_gg_argtopk] 
            sum_sim_topk = torch.sum(sim_gg_topk, dim=-1)
            weight = sim_gg_topk / sum_sim_topk.view(-1, 1)
            
            # Calculate the centroid of the top-k gallery features
            gf_topk = gf_new[sim_gg_argtopk]
            gf_topk_p = gf_topk.permute(0, 2, 1)
            centroid_g = torch.bmm(gf_topk_p, weight.unsqueeze(-1)).squeeze()
            # Store centroid and camera ID's contextual cross-entropy value
            cce[indx] = self.CCE(indx, g_camids)
            dict_umvf[indx] = centroid_g

        del sim_gg
        
        simmat_list = []
        device = gf.device

        # Compute similarity using UFFM + AMC for each query or just using UFFM
        if alpha is not None:
            for i in tqdm(range(len(q_ids)), desc="CalcSimilarity: "):
                curr_camid = q_camids[i]  # Camera ID of the current query
                sim_qg = torch.mm(qf[i].view(1, -1), gf.t())
                centroid_g = dict_umvf[curr_camid]
                sim_centroid = torch.mm(qf[i].view(1, -1), centroid_g.t())
                fusion = alpha * sim_qg + beta * sim_centroid + theta * cce[curr_camid].to(sim_qg.device)
                simmat_list.append(fusion.view(1, -1))
        else:
            for i in tqdm(range(len(q_ids)), desc="CalcSimilarity: "):
                curr_camid = q_camids[i]  # Camera ID of the current query
                centroid_g = dict_umvf[curr_camid]
                sim_centroid = torch.mm(qf[i].view(1, -1), centroid_g.t())
                simmat_list.append(sim_centroid.view(1, -1))

        simmat = torch.cat(simmat_list, dim=0)
        del dict_umvf

        distmat = 1 - simmat

        return distmat.cpu().numpy()
    

    def CCE(self, q_camid, g_camids):
        q_camid = torch.tensor(q_camid)
        g_camids = torch.tensor(g_camids)

        x = torch.tensor(0.0)
        alpha = torch.tensor(1.0, dtype=x.dtype)
        
        cce = torch.where(q_camid == g_camids, alpha, x)
        cce = cce.view(1, -1)

        return cce