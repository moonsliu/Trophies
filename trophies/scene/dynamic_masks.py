"""Frame correspondence and human-mask utilities for scene reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def video_frame_count(video: Optional[Path]) -> int:
    if video is None or not video.exists():
        raise FileNotFoundError(f"Cannot infer frame indices; video not found: {video}")
    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if total <= 0:
        raise ValueError(f"Could not read frame count from {video}")
    return total


def infer_frame_indices(video: Optional[Path], n_views: int) -> np.ndarray:
    """Match the uniform sampling used by scripts/extract_video_frames.py."""

    if video is None:
        raise ValueError("--frame-indices is required when --video is omitted")
    total = video_frame_count(video)
    if n_views == 1:
        return np.array([0], dtype=np.int64)
    last = max(total - 2, 0)
    return np.array(
        [int(round(index * last / (n_views - 1))) for index in range(n_views)],
        dtype=np.int64,
    )


def parse_frame_indices(value: Optional[str], video: Optional[Path], n_views: int) -> np.ndarray:
    if value:
        indices = np.array(
            [int(item.strip()) for item in value.split(",") if item.strip()],
            dtype=np.int64,
        )
    else:
        indices = infer_frame_indices(video, n_views)
    if indices.shape[0] != n_views:
        raise ValueError(f"Expected {n_views} frame indices, got {indices.shape[0]}")
    if (indices < 0).any():
        raise ValueError("Frame indices must be non-negative")
    return indices


def decode_rle_mask(rle: object) -> np.ndarray:
    from pycocotools import mask as mask_utils

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def resize_human_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
    dilation: int,
) -> np.ndarray:
    """Resize and center-crop a mask without changing image geometry."""

    target_h, target_w = target_shape
    source_h, source_w = mask.shape
    scale = max(target_w / float(source_w), target_h / float(source_h))
    resized_w = max(target_w, int(round(source_w * scale)))
    resized_h = max(target_h, int(round(source_h * scale)))
    resized = cv2.resize(
        mask.astype(np.uint8),
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    resized = resized[top : top + target_h, left : left + target_w].astype(bool)
    if resized.shape != (target_h, target_w):
        raise ValueError(
            f"Could not map human mask {mask.shape} to scene mask {target_shape}"
        )
    if dilation > 0:
        kernel_size = 2 * int(dilation) + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        resized = cv2.dilate(
            resized.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
    return resized


def apply_human_masks(
    result: Dict[str, Any],
    masks_path: Optional[Path],
    frame_indices: np.ndarray,
    dilation: int,
) -> Dict[str, Any]:
    """Remove decoded human pixels from per-view DUSt3R validity masks."""

    if masks_path is None:
        return {"human_masks_applied": False}

    masks = np.load(masks_path, allow_pickle=True)
    if frame_indices.max(initial=0) >= len(masks):
        raise ValueError("frame indices reference entries outside masks.npy")

    filtered_masks = []
    human_masks = []
    stats = []
    for view_idx, (scene_mask, frame_idx) in enumerate(
        zip(result["msk"], frame_indices.tolist())
    ):
        scene_mask_bool = np.asarray(scene_mask).astype(bool)
        human_mask = decode_rle_mask(masks[frame_idx])
        human_mask = resize_human_mask(human_mask, scene_mask_bool.shape, dilation)
        filtered = scene_mask_bool & ~human_mask

        before = int(scene_mask_bool.sum())
        removed = int((scene_mask_bool & human_mask).sum())
        stats.append(
            {
                "view_idx": int(view_idx),
                "frame_idx": int(frame_idx),
                "scene_points_before": before,
                "human_mask_pixels": int(human_mask.sum()),
                "scene_points_removed": removed,
                "scene_points_after": int(filtered.sum()),
            }
        )
        filtered_masks.append(filtered.astype(np.bool_))
        human_masks.append(human_mask.astype(np.bool_))

    result["msk"] = filtered_masks
    result["human_msk"] = human_masks
    print(f"Applied human masks from {masks_path}")
    for item in stats:
        print(
            f"  frame {item['frame_idx']:04d}: removed "
            f"{item['scene_points_removed']} / {item['scene_points_before']} points"
        )
    return {
        "human_masks_applied": True,
        "human_masks_path": str(masks_path),
        "human_mask_dilation": int(dilation),
        "human_mask_stats": stats,
    }
