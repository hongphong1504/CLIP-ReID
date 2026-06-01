import json
import torch
import torch.nn as nn
import random
import numpy as np
import os
from config import cfg
import argparse
from data.make_dataloader_myclipreid import make_dataloader
from model.make_model_myclipreid import make_model
from processor.inference import do_inference
from utils.logger import setup_logger

from evaluation.builder import build_evaluator
from evaluation.refiner import AMCEstimator

def set_seed_for_testing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/vit_myclipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed_for_testing(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("MyCLIPReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    logger.info("model: {}".format(model))

    device = cfg.MODEL.DEVICE

    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    alpha = None
    beta = None
    theta = None

    if cfg.TEST.METHOD == "uffm_amc":
        logger.info("Estimating AMC coefficients from training set")

        with open("caption/captions_all_BLIP_finetuned.json", "r") as f:
            captions_all = json.load(f)
    
        caption_dict = {
            item['image_path'] : item['caption'] for item in captions_all if item['status'] == "ok"
        }

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

    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg, dataset_name, is_train=False)

        evaluator = build_evaluator(
            cfg=cfg,
            num_query=num_query,
            alpha=alpha,
            beta=beta,
            theta=theta
        )
        logger.info(f"Evaluating on {dataset_name}")
        do_inference(cfg, model, val_loader, evaluator)
    


