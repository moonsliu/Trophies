"""Selectable camera trajectory back-ends."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from . import masked_droid_slam

__BackendModule = Any


def _build_registry() -> Dict[str, __BackendModule]:
    return {"droid": masked_droid_slam}


_BACKENDS = _build_registry()
_DEFAULT_BACKEND = os.environ.get("TROPHIES_SLAM_BACKEND", "droid").lower()


def get_available_backends() -> Tuple[str, ...]:
    return tuple(sorted(_BACKENDS.keys()))


def _resolve_backend(name: Optional[str]) -> _BackendModule:
    if name is None:
        name = _DEFAULT_BACKEND
    key = name.lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"Unknown SLAM backend '{name}'. Available options: {', '.join(get_available_backends())}"
        )
    return _BACKENDS[key]


def calibrate_intrinsics(
    img_folder: str,
    masks: Optional[torch.Tensor] = None,
    *,
    backend: Optional[str] = None,
    **backend_kwargs: Any,
) -> Tuple[np.ndarray, bool]:
    backend_impl = _resolve_backend(backend)
    return backend_impl.calibrate_intrinsics(img_folder, masks, **backend_kwargs)


def run_metric_slam(
    img_folder: str,
    masks: Optional[torch.Tensor] = None,
    calib: Optional[Sequence[float]] = None,
    *,
    backend: Optional[str] = None,
    **backend_kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backend_impl = _resolve_backend(backend)
    return backend_impl.run_metric_slam(
        img_folder,
        masks=masks,
        calib=calib,
        **backend_kwargs,
    )
