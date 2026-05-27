import logging
from .evaluator import StandardEvaluator, AdvancedEvaluator

logger = logging.getLogger("MyCLIPReID.eval")

SUPPORTED_METHODS = {
    "baseline",
    "rerank",
    "uffm",
    "uffm_amc"
}

def build_evaluator(cfg, num_query, alpha=None, beta=None, theta=None):
    method = cfg.TEST.METHOD.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported TEST.METHOD: {method}. "
            f"Supported methods: {SUPPORTED_METHODS}"
        )
    
    logger.info(f"Building evaluator with method: {method}")

    if method == "baseline":
        return StandardEvaluator(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    return AdvancedEvaluator(cfg=cfg, num_query=num_query, alpha=alpha, beta=beta, theta=theta, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)