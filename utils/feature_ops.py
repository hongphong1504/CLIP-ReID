import torch


def normalize(feats: torch.Tensor):
    return torch.nn.functional.normalize(
        feats,
        dim=1,
        p=2
    )