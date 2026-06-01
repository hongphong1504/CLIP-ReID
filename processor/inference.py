import json
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

    with open("caption/captions_all_BLIP_finetuned.json", "r") as f:
        captions_all = json.load(f)
    
    caption_dict = {
        item['image_path'] : item['caption'] for item in captions_all if item['status'] == "ok"
    }

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