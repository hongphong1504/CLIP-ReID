import torch
import torch.nn as nn
import random
import numpy as np
import os
from config import cfg_base as cfg
import argparse
from data.make_dataloader import make_dataloader
from model.make_model_base import make_model
from processor.inference import do_inference_base as do_inference
from utils.logger import setup_logger

from evaluation.builder import build_evaluator

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
        "--config_file", default="configs/AGReID/vit_base.yml", help="path to config file", type=str
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

    logger.info("Enter inferencing")

    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg, dataset_name, is_train=False)

        evaluator = build_evaluator(
            cfg=cfg,
            num_query=num_query
        )

        logger.info(f"Evaluating on {dataset_name}")
        do_inference(cfg, model, val_loader, evaluator)
    


