#!/usr/bin/env python
"""Check packages needed by Trophies camera and human-motion estimation."""

from __future__ import annotations

import importlib
import sys

REQUIRED = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "pycocotools",
    "pytorch_lightning",
    "segment_anything",
    "detectron2",
    "pytorch3d",
    "supervision",
    "loguru",
    "torchmin",
    "smplx",
    "skimage",
    "timm",
    "lietorch",
    "droid_backends",
    "trophies.camera",
    "trophies.pipeline",
    "trophies.humans",
]

missing = []
for name in REQUIRED:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        missing.append((name, type(exc).__name__, str(exc)))
        print(f"{name}: FAIL ({type(exc).__name__}: {str(exc)[:160]})")
    else:
        version = getattr(module, "__version__", "")
        print(f"{name}: OK {version}")

if missing:
    print("\nMissing or broken camera/human-motion dependencies:")
    for name, kind, message in missing:
        print(f"  - {name}: {kind}: {message[:220]}")
    sys.exit(1)

print("\nTrophies camera/human-motion dependency check passed.")
