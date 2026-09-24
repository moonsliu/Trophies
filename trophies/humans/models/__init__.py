"""Temporal human reconstruction model loader."""
from pathlib import Path
import torch
from trophies.humans.config import get_default_config
from .hmr_vimo import HMR_VIMO


def get_temporal_human_model(checkpoint, device="cuda"):
    model = HMR_VIMO(get_default_config(device))
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("smpl.")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("smpl.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    return model.to(device).eval()


__all__ = ["HMR_VIMO", "get_temporal_human_model"]
