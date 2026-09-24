"""Align a free-camera DUSt3R reconstruction to a reference trajectory."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tyro

from trophies.scene.dynamic_masks import apply_human_masks, parse_frame_indices
from trophies.scene.sim3 import (
    SimilarityTransform,
    apply_similarity_to_points,
    apply_similarity_to_pose,
    estimate_similarity_transform,
    rotation_errors_degrees,
)


@dataclass
class Options:
    scene_pkl: Path = Path("outputs/dust3r_single/scene.pkl")
    """Raw scene.pkl from trophies.scene.dust3r_reconstruct."""

    camera: Path = Path("demo_data/quick_demo/slam/camera.npy")
    """Reference camera.npy containing world_cam_R and world_cam_T."""

    video: Optional[Path] = None
    """Source video used to infer the sampled source-frame indices."""

    frame_indices: Optional[str] = None
    """Comma-separated source-video frame indices. Inferred from --video when omitted."""

    human_masks: Optional[Path] = None
    """COCO-RLE masks.npy. Defaults to camera.parent / masks.npy."""

    mask_humans: bool = True
    """Remove human pixels from the aligned static-scene masks."""

    human_mask_dilation: int = 7
    """Dilation radius in resized DUSt3R pixels."""

    allow_scaling: bool = True
    """Estimate uniform scale in addition to rotation and translation."""

    output_dir: Path = Path("outputs/dust3r_single_sim3")
    """Directory for the aligned scene."""

    scene_name: str = "scene"
    """Output pickle stem."""


def _load_camera(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Camera file not found: {path}")
    loaded = np.load(path, allow_pickle=True)
    camera = loaded.item() if not isinstance(loaded, np.lib.npyio.NpzFile) else {key: loaded[key] for key in loaded.files}
    for key in ("world_cam_R", "world_cam_T"):
        if key not in camera:
            raise KeyError(f"{path} does not contain {key}")
    return camera


def _ordered_entries(payload: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    env = payload.get("dust3r_ga_output")
    if not isinstance(env, dict) or not env:
        raise ValueError("dust3r_ga_output is missing or empty in the input scene")

    source_images = payload.get("source_images")
    if not source_images:
        return list(env.items())

    ordered: List[Tuple[str, Dict[str, Any]]] = []
    used = set()
    for source in source_images:
        key = Path(source).stem
        if key not in env:
            raise KeyError(f"Scene entry '{key}' for source image {source} is missing")
        if key in used:
            raise ValueError(f"Duplicate source-image stem '{key}' cannot be aligned unambiguously")
        ordered.append((key, env[key]))
        used.add(key)
    if len(ordered) != len(env):
        extra = sorted(set(env).difference(used))
        raise ValueError(f"Scene has entries not represented by source_images: {extra}")
    return ordered


def _preserve_dtype(values: np.ndarray, reference: Any) -> np.ndarray:
    dtype = np.asarray(reference).dtype
    return np.asarray(values).astype(dtype, copy=False)


def _transform_entry(entry: Dict[str, Any], transform: SimilarityTransform) -> None:
    raw_pose = np.asarray(entry["cam2world"])
    raw_points = np.asarray(entry["pts3d"])
    entry["cam2world_dust3r"] = raw_pose.copy()
    entry["cam2world"] = _preserve_dtype(apply_similarity_to_pose(raw_pose, transform), raw_pose)
    entry["pts3d"] = _preserve_dtype(apply_similarity_to_points(raw_points, transform), raw_points)
    if entry.get("depths") is not None:
        raw_depths = np.asarray(entry["depths"])
        entry["depths"] = _preserve_dtype(raw_depths * transform.scale, raw_depths)


def _mask_path(opts: Options) -> Optional[Path]:
    if not opts.mask_humans:
        return None
    path = opts.human_masks or (opts.camera.parent / "masks.npy")
    if not path.is_file():
        print(f"Human mask removal requested, but masks were not found: {path}")
        return None
    return path


def align_scene(opts: Options) -> Path:
    if not opts.scene_pkl.is_file():
        raise FileNotFoundError(f"DUSt3R scene not found: {opts.scene_pkl}")
    with opts.scene_pkl.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary in {opts.scene_pkl}")

    ordered = _ordered_entries(payload)
    if len(ordered) < 3:
        raise ValueError("At least three DUSt3R cameras are required for Sim(3) alignment")

    camera = _load_camera(opts.camera)
    frame_indices = parse_frame_indices(opts.frame_indices, opts.video, len(ordered))
    world_r = np.asarray(camera["world_cam_R"], dtype=np.float64)
    world_t = np.asarray(camera["world_cam_T"], dtype=np.float64)
    if frame_indices.max(initial=0) >= len(world_t):
        raise ValueError("frame indices reference entries outside camera.npy")

    dust3r_poses = np.stack([np.asarray(entry["cam2world"], dtype=np.float64) for _, entry in ordered])
    dust3r_centers = dust3r_poses[:, :3, 3]
    target_centers = world_t[frame_indices]
    transform = estimate_similarity_transform(
        dust3r_centers,
        target_centers,
        allow_scaling=opts.allow_scaling,
    )

    aligned_centers = apply_similarity_to_points(dust3r_centers, transform)
    center_errors = np.linalg.norm(aligned_centers - target_centers, axis=1)
    aligned_rotations = np.einsum("ij,njk->nik", transform.rotation, dust3r_poses[:, :3, :3])
    target_rotations = world_r[frame_indices]
    rotation_errors = rotation_errors_degrees(aligned_rotations, target_rotations)

    for _, entry in ordered:
        _transform_entry(entry, transform)

    mask_result = {"msk": [entry.get("msk") for _, entry in ordered]}
    mask_metadata = apply_human_masks(
        mask_result,
        _mask_path(opts),
        frame_indices,
        opts.human_mask_dilation,
    )
    if mask_metadata["human_masks_applied"]:
        for idx, (_, entry) in enumerate(ordered):
            entry["msk"] = mask_result["msk"][idx]
            entry["human_msk"] = mask_result["human_msk"][idx]

    alignment = {
        "type": "dust3r_to_reference_camera_sim3",
        "convention": "p_world = scale * (rotation @ p_dust3r) + translation",
        **transform.to_dict(),
        "source_scene": str(opts.scene_pkl),
        "reference_camera": str(opts.camera),
        "frame_indices": frame_indices.tolist(),
        "dust3r_camera_centers": dust3r_centers.tolist(),
        "reference_camera_centers": target_centers.tolist(),
        "aligned_camera_centers": aligned_centers.tolist(),
        "camera_center_errors": center_errors.tolist(),
        "camera_center_rmse": float(np.sqrt(np.mean(np.square(center_errors)))),
        "camera_center_max_error": float(center_errors.max()),
        "camera_rotation_errors_degrees": rotation_errors.tolist(),
        "camera_rotation_mean_degrees": float(rotation_errors.mean()),
        "camera_rotation_max_degrees": float(rotation_errors.max()),
    }
    payload["alignment"] = alignment
    payload["reference_camera"] = {
        "world_cam_R": target_rotations.astype(np.float32),
        "world_cam_T": target_centers.astype(np.float32),
        "frame_indices": frame_indices.astype(np.int64),
    }
    settings = dict(payload.get("settings", {}))
    settings.update(
        {
            "sim3_alignment": True,
            "sim3_source_scene": str(opts.scene_pkl),
            "sim3_reference_camera": str(opts.camera),
            "sim3_allow_scaling": bool(opts.allow_scaling),
            "frame_indices": frame_indices.tolist(),
            **mask_metadata,
        }
    )
    payload["settings"] = settings

    opts.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = opts.output_dir / f"{opts.scene_name}.pkl"
    if output_path.resolve() == opts.scene_pkl.resolve():
        raise ValueError("Output scene must differ from the raw DUSt3R input scene")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(output_path)

    print(f"Aligned DUSt3R scene written to {output_path}")
    print(f"frame_indices={frame_indices.tolist()}")
    print(f"scale={transform.scale:.8f}")
    print(f"camera_center_rmse={alignment['camera_center_rmse']:.6f} m")
    print(f"camera_center_max_error={alignment['camera_center_max_error']:.6f} m")
    print(f"camera_rotation_mean_error={alignment['camera_rotation_mean_degrees']:.3f} deg")
    return output_path


def main() -> None:
    align_scene(tyro.cli(Options))


if __name__ == "__main__":
    main()
