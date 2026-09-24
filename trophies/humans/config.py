"""Inference configuration for the temporal human estimator."""
from types import SimpleNamespace


def get_default_config(device: str = "cuda") -> SimpleNamespace:
    return SimpleNamespace(
        IMG_RES=256,
        device=device,
        MODEL=SimpleNamespace(
            ST_MODULE=True,
            MOTION_MODULE=True,
            ST_HDIM=512,
            MOTION_HDIM=384,
            ST_NLAYER=6,
            MOTION_NLAYER=6,
        ),
    )
