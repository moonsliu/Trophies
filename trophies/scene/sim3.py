"""Similarity transforms used to align DUSt3R geometry to a world frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    """Seven degree-of-freedom transform ``dst = scale * R @ src + t``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "matrix": self.as_matrix().tolist(),
        }


def _validate_points(name: str, points: np.ndarray, minimum: int = 1) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {values.shape}")
    if len(values) < minimum:
        raise ValueError(f"{name} needs at least {minimum} points, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def estimate_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_scaling: bool = True,
    eps: float = 1e-12,
) -> SimilarityTransform:
    """Estimate the Umeyama Sim(3) that maps source points to target points."""

    source = _validate_points("source", source, minimum=3)
    target = _validate_points("target", target, minimum=3)
    if source.shape != target.shape:
        raise ValueError(f"source and target must have the same shape, got {source.shape} and {target.shape}")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    source_variance = float(np.square(source_centered).sum() / len(source))
    if source_variance <= eps:
        raise ValueError("Source camera centers have near-zero variance")

    covariance = (target_centered.T @ source_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    if allow_scaling:
        scale = float(np.dot(singular_values, np.diag(correction)) / source_variance)
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= eps:
        raise ValueError(f"Estimated an invalid similarity scale: {scale}")

    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def apply_similarity_to_points(points: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    """Apply a Sim(3) to points with any leading dimensions ending in XYZ."""

    values = np.asarray(points)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"points must end in dimension 3, got {values.shape}")
    flat = values.reshape(-1, 3).astype(np.float64, copy=False)
    aligned = transform.scale * (flat @ transform.rotation.T) + transform.translation
    return aligned.reshape(values.shape)


def apply_similarity_to_pose(pose: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    """Apply a world-frame Sim(3) to a camera-to-world pose."""

    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"pose must have shape [4, 4], got {value.shape}")
    aligned = value.copy()
    aligned[:3, :3] = transform.rotation @ value[:3, :3]
    aligned[:3, 3] = apply_similarity_to_points(value[None, :3, 3], transform)[0]
    aligned[3] = np.array([0.0, 0.0, 0.0, 1.0])
    return aligned


def rotation_errors_degrees(estimated: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return geodesic SO(3) errors for corresponding rotation matrices."""

    estimated = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if estimated.shape != target.shape or estimated.ndim != 3 or estimated.shape[1:] != (3, 3):
        raise ValueError("estimated and target rotations must both have shape [N, 3, 3]")
    relative = np.einsum("nij,njk->nik", np.transpose(estimated, (0, 2, 1)), target)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))
