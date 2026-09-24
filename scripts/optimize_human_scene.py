#!/usr/bin/env python
"""Refine SMPL tracks using contact with a reconstructed floor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from trophies.optimization.ground_contact import optimize_ground_contact


@dataclass
class Options:
    scene_pkl: Path = Path("outputs/dust3r_single_sim3/scene.pkl")
    """Free-camera DUSt3R scene aligned to the Trophies world frame."""

    human_motion_dir: Path = Path("demo_data/quick_demo/slam")
    """Directory containing camera.npy and hps/hps_track_*.npy."""

    output_dir: Path = Path("outputs/single_human_scene")
    """Destination containing camera.npy, optimized hps/, and ground_contact.json."""

    smpl_model_dir: Path = Path("body_models/smpl")
    """Directory containing the licensed neutral SMPL model."""

    device: str = "cuda"
    confidence_threshold: float = 0.0
    floor_distance_threshold: float = 0.03
    floor_angle_threshold_degrees: float = 15.0
    floor_max_planes: int = 6
    floor_min_inliers: int = 800
    floor_max_points: int = 80_000
    floor_ransac_iterations: int = 400
    contact_margin: float = 0.04
    penetration_margin: float = 0.002
    softmin_temperature: float = 50.0
    regularization_weight: float = 1e-4
    learning_rate: float = 0.05
    iterations: int = 300
    max_offset: float = 1.5
    seed: int = 2026


def main() -> None:
    options = tyro.cli(Options)
    optimize_ground_contact(
        scene_path=options.scene_pkl,
        human_motion_dir=options.human_motion_dir,
        output_dir=options.output_dir,
        smpl_model_dir=options.smpl_model_dir,
        device=options.device,
        confidence_threshold=options.confidence_threshold,
        floor_distance_threshold=options.floor_distance_threshold,
        floor_angle_threshold_degrees=options.floor_angle_threshold_degrees,
        floor_max_planes=options.floor_max_planes,
        floor_min_inliers=options.floor_min_inliers,
        floor_max_points=options.floor_max_points,
        floor_ransac_iterations=options.floor_ransac_iterations,
        contact_margin=options.contact_margin,
        penetration_margin=options.penetration_margin,
        softmin_temperature=options.softmin_temperature,
        regularization_weight=options.regularization_weight,
        learning_rate=options.learning_rate,
        iterations=options.iterations,
        max_offset=options.max_offset,
        seed=options.seed,
    )


if __name__ == "__main__":
    main()
