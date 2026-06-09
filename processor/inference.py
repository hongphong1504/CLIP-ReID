import logging
import torch
import torch.nn as nn

logger = logging.getLogger("MyCLIPReID.test")

@torch.no_grad()
def do_inference(cfg,
                 model,
                 val_loader,
                 evaluator):
    device = cfg.MODEL.DEVICE

    for n_iter, (img, pid, camid, camids, target_view, captions, img_path) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            captions = captions.to(device)
            
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


@torch.no_grad()
def do_inference_base(cfg,
                 model,
                 val_loader,
                 evaluator):
    device = cfg.MODEL.DEVICE

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    
    model.eval()

    for n_iter, (img, pid, camid, camids, target_view, img_path) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None

            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update(feats=feat, pids=pid, camids=camid)

    cmc, mAP = evaluator.evaluate()

    logger.info("Validation Results ")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]