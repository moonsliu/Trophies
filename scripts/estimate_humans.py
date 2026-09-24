#!/usr/bin/env python
"""Estimate temporally coherent SMPL parameters from tracked video people."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gdown
import numpy as np
import torch

from trophies.humans import get_temporal_human_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "vimo_checkpoint.pth.tar"
VIMO_CHECKPOINT_URL = "https://drive.google.com/file/d/1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW/view?usp=share_link"
VIMO_CHECKPOINT_SHA256 = "197c333255dced98f48b67c0c2f6c630a5d1fe246fa9131f95aa4f15fb4080e0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_name(f".{checkpoint.name}.part")
    partial.unlink(missing_ok=True)
    print("Downloading the VIMO checkpoint provided by TRAM for human-motion reconstruction:")
    print(f"  {VIMO_CHECKPOINT_URL}")
    try:
        downloaded = gdown.download(VIMO_CHECKPOINT_URL, str(partial), quiet=False, fuzzy=True)
        if downloaded is None or not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("VIMO checkpoint download did not produce a file")
        digest = file_sha256(partial)
        if digest != VIMO_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"VIMO checkpoint checksum mismatch: expected {VIMO_CHECKPOINT_SHA256}, got {digest}"
            )
        partial.replace(checkpoint)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"VIMO checkpoint ready: {checkpoint}")
    return checkpoint



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None, help="input video used to derive the default result directory")
    parser.add_argument("--input_dir", type=Path, default=None, help="camera-stage directory containing images/, camera.npy, and tracks.npy")
    parser.add_argument("--output_dir", type=Path, default=None, help="output directory (default: input directory)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="temporal human model checkpoint (TRAM VIMO is downloaded automatically when missing)")
    parser.add_argument("--max_humans", type=int, default=20, help="maximum number of tracks to reconstruct")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return parser.parse_args()


def resolve_directories(args):
    input_dir = args.input_dir
    if input_dir is None:
        if args.video is None:
            raise ValueError("Provide --input_dir or --video")
        input_dir = Path("results") / args.video.stem
    return input_dir.resolve(), (args.output_dir or input_dir).resolve()


def load_inputs(input_dir):
    image_dir = input_dir / "images"
    image_files = np.array(sorted(image_dir.glob("*.jpg")))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No JPG frames found under {image_dir}")
    camera_path = input_dir / "camera.npy"
    tracks_path = input_dir / "tracks.npy"
    if not camera_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError(f"Expected camera.npy and tracks.npy under {input_dir}. Run estimate_camera.py first.")
    camera = np.load(camera_path, allow_pickle=True).item()
    tracks = np.load(tracks_path, allow_pickle=True).item()
    return image_files, camera, tracks


def main():
    args = parse_args()
    if args.max_humans < 1:
        raise ValueError("--max_humans must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    input_dir, output_dir = resolve_directories(args)
    image_files, camera, tracks = load_inputs(input_dir)
    checkpoint = ensure_checkpoint(args.checkpoint)

    output_hps = output_dir / "hps"
    output_hps.mkdir(parents=True, exist_ok=True)
    model = get_temporal_human_model(checkpoint, device=args.device)
    ordered_tracks = sorted(tracks.items(), key=lambda item: len(item[1]), reverse=True)

    written = 0
    for source_track_id, track in ordered_tracks:
        valid = np.asarray([entry["det"] for entry in track], dtype=bool)
        boxes = np.concatenate([entry["det_box"] for entry in track])
        frames = np.asarray([entry["frame"] for entry in track], dtype=np.int64)
        result = model.inference(
            image_files,
            boxes,
            valid=valid,
            frame=frames,
            img_focal=camera["img_focal"],
            img_center=camera["img_center"],
            device=args.device,
        )
        if result is None:
            print(f"Skipping track {source_track_id}: no contiguous segment has at least 16 detections")
            continue
        result["source_track_id"] = int(source_track_id)
        destination = output_hps / f"hps_track_{written}.npy"
        np.save(destination, result)
        frame_count = len(result["frame"])
        print(f"Saved {destination} ({frame_count} frames from source track {source_track_id})")
        written += 1
        if written >= args.max_humans:
            break

    if written == 0:
        raise RuntimeError("No human track produced a valid 16-frame reconstruction")
    print(f"Human reconstruction complete: {written} track(s) written to {output_hps}")


if __name__ == "__main__":
    main()
