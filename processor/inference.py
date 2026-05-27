import json
import logging
import torch
import torch.nn as nn

from evaluation.builder import build_evaluator
from evaluation.refiner import AMCEstimator

logger = logging.getLogger("MyCLIPReID.test")

@torch.no_grad()
def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 train_loader=None):
    device = cfg.MODEL.DEVICE

    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    with open("caption/captions_all_BLIP_finetuned.json", "r") as f:
        captions_all = json.load(f)
    
    caption_dict = {
        item['image_path'] : item['caption'] for item in captions_all if item['status'] == "ok"
    }

    method = cfg.TEST.METHOD

    alpha = None
    beta = None
    theta = None

    if method == "uffm_amc":

        assert train_loader is not None, ("train_loader must be provided when AMC is enabled")

        logger.info("Estimating AMC coefficients from training set")

        estimator = AMCEstimator(cfg)

        alpha, beta, theta = estimator.fit(
            model=model,
            train_loader=train_loader,
            caption_dict=caption_dict,
            n_data=cfg.TEST.AMC_N_TRIPLETS,
            rand_seed=cfg.SOLVER.SEED
        )

        logger.info(
            f"AMC coefficients | "
            f"alpha={alpha:.4f}, "
            f"beta={beta:.4f}, "
            f"theta={theta:.4f}"
        )

    evaluator = build_evaluator(
            cfg=cfg,
            num_query=num_query,
            alpha=alpha,
            beta=beta,
            theta=theta
        )

    for n_iter, (img, pid, camid, camids, target_view, img_path) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            captions = [caption_dict[p] for p in img_path]

            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None

            feat = model(image=img, caption=captions, cam_label=camids, view_label=target_view)
            evaluator.update(feats=feat, pids=pid, camids=camid)

    cmc, mAP = evaluator.evaluate()

    logger.info("Validation Results ")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]