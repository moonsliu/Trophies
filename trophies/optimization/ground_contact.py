"""Refine SMPL tracks against a floor plane estimated from a scene point cloud."""

from __future__ import annotations

import json
import math
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# Indices 7 and 8 are the SMPL ankles; 29 through 34 are the six foot
# landmarks appended by smplx's vertex joint selector.
SMPL_FOOT_JOINT_INDICES = (7, 31, 29, 30, 8, 34, 32, 33)


@dataclass
class FloorPlane:
    """Plane represented by ``normal @ point + offset = 0``."""

    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    inlier_count: int
    tilt_degrees: float
    height: float

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["normal"] = self.normal.tolist()
        data["centroid"] = self.centroid.tolist()
        return data


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_scene_points(scene_path: Path, confidence_threshold: float = 0.0) -> np.ndarray:
    """Load valid, background-only points from a Trophies DUSt3R scene pickle."""

    with scene_path.open("rb") as handle:
        payload = pickle.load(handle)
    entries = payload.get("dust3r_ga_output")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"dust3r_ga_output is missing or empty in {scene_path}")

    clouds: List[np.ndarray] = []
    for entry in entries.values():
        points = np.asarray(entry["pts3d"], dtype=np.float32).reshape(-1, 3)
        keep = np.isfinite(points).all(axis=1)
        if entry.get("msk") is not None:
            keep &= np.asarray(entry["msk"]).reshape(-1).astype(bool)
        if entry.get("conf") is not None:
            keep &= np.asarray(entry["conf"]).reshape(-1) > confidence_threshold
        if keep.any():
            clouds.append(points[keep])
    if not clouds:
        raise ValueError(f"No valid scene points found in {scene_path}")
    return np.concatenate(clouds, axis=0)


def _fit_plane(points: np.ndarray, up_axis: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= max(np.linalg.norm(normal), 1e-12)
    if float(normal @ up_axis) < 0.0:
        normal = -normal
    offset = -float(normal @ centroid)
    return normal, offset, centroid


def detect_floor_plane(
    points: np.ndarray,
    *,
    up_axis: Sequence[float] = (0.0, 1.0, 0.0),
    distance_threshold: float = 0.03,
    angle_threshold_degrees: float = 15.0,
    max_planes: int = 6,
    min_inliers: int = 800,
    max_points: int = 80_000,
    ransac_iterations: int = 400,
    seed: int = 2026,
) -> FloorPlane:
    """Detect the lowest sufficiently large near-horizontal plane with RANSAC."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < max(3, min_inliers):
        raise ValueError(f"Need at least {max(3, min_inliers)} finite scene points, got {len(points)}")

    up = np.asarray(up_axis, dtype=np.float64)
    up /= max(np.linalg.norm(up), 1e-12)
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        points = points[rng.choice(len(points), size=max_points, replace=False)]

    active = points.copy()
    candidates: List[FloorPlane] = []
    for _ in range(max(1, max_planes)):
        if len(active) < max(3, min_inliers):
            break
        best_mask: Optional[np.ndarray] = None
        best_count = 0
        for _ in range(max(1, ransac_iterations)):
            sample = active[rng.choice(len(active), size=3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm
            offset = -float(normal @ sample[0])
            mask = np.abs(active @ normal + offset) <= distance_threshold
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_mask = mask

        if best_mask is None or best_count < min_inliers:
            break

        normal, offset, centroid = _fit_plane(active[best_mask], up)
        refined_mask = np.abs(active @ normal + offset) <= distance_threshold
        normal, offset, centroid = _fit_plane(active[refined_mask], up)
        refined_mask = np.abs(active @ normal + offset) <= distance_threshold
        inlier_count = int(refined_mask.sum())
        cosine = float(np.clip(normal @ up, -1.0, 1.0))
        tilt = float(math.degrees(math.acos(cosine)))
        if inlier_count >= min_inliers and tilt <= angle_threshold_degrees:
            candidates.append(
                FloorPlane(
                    normal=normal.astype(np.float32),
                    offset=float(offset),
                    centroid=centroid.astype(np.float32),
                    inlier_count=inlier_count,
                    tilt_degrees=tilt,
                    height=float(centroid @ up),
                )
            )
        active = active[~refined_mask]

    if not candidates:
        raise RuntimeError(
            "Could not detect a near-horizontal floor plane. "
            "Try increasing --floor-distance-threshold or --floor-angle-threshold-degrees."
        )

    largest = max(candidate.inlier_count for candidate in candidates)
    credible = [candidate for candidate in candidates if candidate.inlier_count >= max(min_inliers, largest // 4)]
    return min(credible, key=lambda candidate: candidate.height)


def soft_min(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Differentiable minimum as a softmax-weighted average."""

    temperature = max(float(temperature), 1e-3)
    weights = torch.softmax(-temperature * values, dim=-1)
    return (weights * values).sum(dim=-1)


def optimize_normal_offset(
    signed_foot_distances: torch.Tensor,
    *,
    contact_margin: float = 0.04,
    penetration_margin: float = 0.002,
    softmin_temperature: float = 50.0,
    regularization_weight: float = 1e-4,
    learning_rate: float = 0.05,
    iterations: int = 300,
    max_offset: float = 1.5,
) -> Tuple[float, List[float]]:
    """Optimize one shared track offset along the floor normal."""

    if signed_foot_distances.ndim != 2 or signed_foot_distances.numel() == 0:
        raise ValueError("signed_foot_distances must have shape [frames, foot_joints]")

    offset = torch.nn.Parameter(torch.zeros((), device=signed_foot_distances.device))
    optimizer = torch.optim.Adam([offset], lr=learning_rate)
    history: List[float] = []

    for _ in range(max(1, iterations)):
        optimizer.zero_grad()
        minimum = soft_min(signed_foot_distances + offset, softmin_temperature)
        penetration = torch.relu(-minimum - penetration_margin).square()
        hovering = torch.relu(minimum - contact_margin).square()
        loss = (penetration + hovering).mean() + regularization_weight * offset.square()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            offset.clamp_(-max_offset, max_offset)
        history.append(float(loss.detach().cpu()))

    return float(offset.detach().cpu()), history


def world_offset_to_camera(
    world_offset: torch.Tensor,
    world_cam_rotations: torch.Tensor,
) -> torch.Tensor:
    """Convert a world-space vector to each frame's camera coordinates."""

    return torch.einsum("bij,j->bi", world_cam_rotations.transpose(1, 2), world_offset)


def _load_track(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return {key: data[key] for key in data.files}
    value = data.item()
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value


def _save_like(original: Any, values: torch.Tensor) -> Any:
    values = values.detach().cpu()
    if isinstance(original, torch.Tensor):
        return values.to(dtype=original.dtype)
    return values.numpy().astype(np.asarray(original).dtype, copy=False)


def _track_foot_distances(
    track: Dict[str, Any],
    smpl: Any,
    world_cam_r: torch.Tensor,
    world_cam_t: torch.Tensor,
    plane: FloorPlane,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {"pred_rotmat", "pred_shape", "pred_trans", "frame"}
    missing = sorted(required.difference(track))
    if missing:
        raise KeyError(f"Track is missing required fields: {missing}")

    frames = torch.as_tensor(_as_numpy(track["frame"]), dtype=torch.long, device=device).reshape(-1)
    valid = (frames >= 0) & (frames < len(world_cam_t))
    if not valid.any():
        raise ValueError("Track has no frames covered by camera.npy")

    rotations = torch.as_tensor(_as_numpy(track["pred_rotmat"]), dtype=torch.float32, device=device)[valid]
    shapes_all = torch.as_tensor(_as_numpy(track["pred_shape"]), dtype=torch.float32, device=device)
    shapes = shapes_all.mean(dim=0, keepdim=True).repeat(int(valid.sum().item()), 1)
    translations = torch.as_tensor(_as_numpy(track["pred_trans"]), dtype=torch.float32, device=device).reshape(-1, 3)[valid]
    valid_frames = frames[valid]

    with torch.no_grad():
        output = smpl(
            body_pose=rotations[:, 1:],
            global_orient=rotations[:, [0]],
            betas=shapes,
            transl=translations,
            pose2rot=False,
        )
        indices = torch.tensor(SMPL_FOOT_JOINT_INDICES, dtype=torch.long, device=device)
        feet_camera = output.joints.index_select(1, indices)
        cam_r = world_cam_r.index_select(0, valid_frames)
        cam_t = world_cam_t.index_select(0, valid_frames)
        feet_world = torch.einsum("bij,bnj->bni", cam_r, feet_camera) + cam_t[:, None]
        normal = torch.as_tensor(plane.normal, dtype=torch.float32, device=device)
        distances = torch.einsum("bnj,j->bn", feet_world, normal) + float(plane.offset)

    return distances, valid, frames


def optimize_ground_contact(
    *,
    scene_path: Path,
    human_motion_dir: Path,
    output_dir: Path,
    smpl_model_dir: Path,
    device: str = "cuda",
    confidence_threshold: float = 0.0,
    floor_distance_threshold: float = 0.03,
    floor_angle_threshold_degrees: float = 15.0,
    floor_max_planes: int = 6,
    floor_min_inliers: int = 800,
    floor_max_points: int = 80_000,
    floor_ransac_iterations: int = 400,
    contact_margin: float = 0.04,
    penetration_margin: float = 0.002,
    softmin_temperature: float = 50.0,
    regularization_weight: float = 1e-4,
    learning_rate: float = 0.05,
    iterations: int = 300,
    max_offset: float = 1.5,
    seed: int = 2026,
) -> Path:
    """Estimate a floor and save ground-contact-refined copies of all SMPL tracks."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch_device = torch.device(device)

    scene_points = load_scene_points(scene_path, confidence_threshold)
    plane = detect_floor_plane(
        scene_points,
        distance_threshold=floor_distance_threshold,
        angle_threshold_degrees=floor_angle_threshold_degrees,
        max_planes=floor_max_planes,
        min_inliers=floor_min_inliers,
        max_points=floor_max_points,
        ransac_iterations=floor_ransac_iterations,
        seed=seed,
    )

    camera_path = human_motion_dir / "camera.npy"
    if not camera_path.is_file():
        raise FileNotFoundError(f"Missing camera file: {camera_path}")
    camera = np.load(camera_path, allow_pickle=True).item()
    world_cam_r = torch.as_tensor(camera["world_cam_R"], dtype=torch.float32, device=torch_device)
    world_cam_t = torch.as_tensor(camera["world_cam_T"], dtype=torch.float32, device=torch_device)

    track_paths = sorted((human_motion_dir / "hps").glob("hps_track_*.npy"))
    if not track_paths:
        raise FileNotFoundError(f"No hps_track_*.npy files found under {human_motion_dir / 'hps'}")

    import smplx

    smpl = smplx.SMPL(model_path=str(smpl_model_dir), gender="neutral", batch_size=1).to(torch_device).eval()
    normal = torch.as_tensor(plane.normal, dtype=torch.float32, device=torch_device)
    output_hps = output_dir / "hps"
    output_hps.mkdir(parents=True, exist_ok=True)
    shutil.copy2(camera_path, output_dir / "camera.npy")

    track_reports: List[Dict[str, Any]] = []
    for track_path in track_paths:
        track = _load_track(track_path)
        distances, valid, frames = _track_foot_distances(
            track, smpl, world_cam_r, world_cam_t, plane, torch_device
        )
        offset_scalar, history = optimize_normal_offset(
            distances,
            contact_margin=contact_margin,
            penetration_margin=penetration_margin,
            softmin_temperature=softmin_temperature,
            regularization_weight=regularization_weight,
            learning_rate=learning_rate,
            iterations=iterations,
            max_offset=max_offset,
        )

        world_offset = normal * offset_scalar
        valid_frames = frames[valid]
        camera_offsets = world_offset_to_camera(world_offset, world_cam_r.index_select(0, valid_frames))
        original_trans = torch.as_tensor(
            _as_numpy(track["pred_trans"]), dtype=torch.float32, device=torch_device
        ).reshape(-1, 3)
        optimized_trans = original_trans.clone()
        optimized_trans[valid] += camera_offsets
        optimized_trans = optimized_trans.reshape(_as_numpy(track["pred_trans"]).shape)

        output_track = dict(track)
        output_track["pred_trans"] = _save_like(track["pred_trans"], optimized_trans)
        output_track["ground_offset_world"] = world_offset.detach().cpu().numpy().astype(np.float32)
        np.save(output_hps / track_path.name, output_track, allow_pickle=True)

        before = soft_min(distances, softmin_temperature)
        after = soft_min(distances + offset_scalar, softmin_temperature)
        report = {
            "track": track_path.name,
            "frames": int(valid.sum().item()),
            "offset_along_normal": offset_scalar,
            "world_offset": world_offset.detach().cpu().tolist(),
            "contact_distance_before_mean": float(before.mean().cpu()),
            "contact_distance_after_mean": float(after.mean().cpu()),
            "loss_initial": history[0],
            "loss_final": history[-1],
        }
        track_reports.append(report)
        print(
            f"{track_path.name}: offset={offset_scalar:+.4f} m, "
            f"contact={report['contact_distance_before_mean']:+.4f} -> "
            f"{report['contact_distance_after_mean']:+.4f} m, "
            f"loss={history[0]:.6f} -> {history[-1]:.6f}"
        )

    metadata = {
        "scene": str(scene_path),
        "human_motion_input": str(human_motion_dir),
        "floor_plane": plane.to_json(),
        "settings": {
            "contact_margin": contact_margin,
            "penetration_margin": penetration_margin,
            "softmin_temperature": softmin_temperature,
            "regularization_weight": regularization_weight,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "max_offset": max_offset,
        },
        "tracks": track_reports,
    }
    metadata_path = output_dir / "ground_contact.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        "Floor plane: "
        f"normal={plane.normal.tolist()}, offset={plane.offset:.4f}, "
        f"tilt={plane.tilt_degrees:.2f} deg, inliers={plane.inlier_count}"
    )
    print(f"Ground-contact-refined motion written to {output_dir}")
    return output_dir
