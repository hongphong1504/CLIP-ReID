import logging
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn

from torch import amp

from utils.meter import AverageMeter

from evaluation.evaluator import StandardEvaluator

logger = logging.getLogger("MyCLIPReID.train")


def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank
):
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = (
        cfg.SOLVER.CHECKPOINT_PERIOD
    )
    eval_period = cfg.SOLVER.EVAL_PERIOD

    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        num_classes = model.module.num_classes
    else:
        num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    scaler = amp.GradScaler("cuda")

    all_start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None

            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            with amp.autocast("cuda", enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)

            scaler.update()

            with torch.no_grad():
                acc = (score[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    f"Epoch[{epoch}] "
                    f"Iter[{n_iter + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.3f}, "
                    f"Acc: {acc_meter.avg:.3f}, "
                    f"Lr: "
                    f"{scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        epoch_end_time = time.time()

        time_per_batch = (epoch_end_time - epoch_start_time) / (n_iter + 1)

        logger.info(
            f"Epoch {epoch} done. "
            f"Time per batch: "
            f"{time_per_batch:.3f}[s] "
            f"Speed: "
            f"{train_loader.batch_size / time_per_batch:.1f}"
            f"[samples/s]"
        )

        if epoch % checkpoint_period == 0:
            save_checkpoint(cfg, model, epoch)

        if epoch % eval_period == 0:
            validate(cfg=cfg, model=model, val_loader=val_loader, num_query=num_query)

    total_time = timedelta(seconds=time.monotonic() - all_start_time)

    logger.info(f"Total running time: {total_time}")

    logger.info(f"Outputs saved to: {cfg.OUTPUT_DIR}")

def save_checkpoint(
    cfg,
    model,
    epoch
):
    if (cfg.MODEL.DIST_TRAIN and dist.get_rank() != 0):
        return
    save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Checkpoint saved: {save_path}")

@torch.no_grad()
def validate(
    cfg,
    model,
    val_loader,
    num_query
):
    device = cfg.MODEL.DEVICE
    model.eval()
    evaluator = StandardEvaluator(num_query=num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    for (img, vid, camid, camids, target_view, _) in val_loader:
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
        evaluator.update(feats=feat, pids=vid, camids=camid)

    cmc, mAP = evaluator.evaluate()
    logger.info("Validation Results ")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))

    torch.cuda.empty_cache()
