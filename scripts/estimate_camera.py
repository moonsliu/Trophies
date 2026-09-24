import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import hashlib
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as masktool

from trophies.pipeline import detect_segment_track, video2frames
from trophies.camera import (
    run_metric_slam,
    calibrate_intrinsics,
    align_cam_to_world,
    get_available_backends,
)


def _color_from_track_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic but vivid BGR color for a given track id."""
    digest = hashlib.sha1(str(track_id).encode("utf-8")).digest()
    return tuple(55 + (digest[i] % 200) for i in range(3))


def _extract_track_box(entry: dict) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Prefer detection box, otherwise fall back to segmentation box."""
    if entry.get("det") and entry.get("det_box") is not None:
        arr = np.array(entry["det_box"]).reshape(-1)
        if arr.size >= 4:
            coords = arr[:4].astype(float)
            score = float(arr[4]) if arr.size >= 5 else None
            return coords, score

    if entry.get("seg_box") is not None:
        arr = np.array(entry["seg_box"]).reshape(-1)
        if arr.size >= 4:
            coords = arr[:4].astype(float)
            return coords, None

    return None, None


def visualize_estimate_results(seq_folder: str,
                               video_path: str,
                               imgfiles: List[str],
                               boxes: np.ndarray,
                               masks: np.ndarray,
                               tracks: np.ndarray,
                               camera: Dict,
                               stride: int = 1,
                               track_tail: int = 60) -> None:
    """Create debug visualizations for mask, camera, boxes, and tracks."""
    stride = max(1, int(stride))
    os.makedirs(seq_folder, exist_ok=True)
    vis_root = os.path.join(seq_folder, "visualizations")
    frame_dir = os.path.join(vis_root, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Prepare track data for quick lookup per-frame
    track_dict = tracks.item() if isinstance(tracks, np.ndarray) else tracks
    frame_tracks: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for tid, entries in (track_dict or {}).items():
        if not entries:
            continue
        for entry in entries:
            frame_idx = int(entry.get("frame", -1))
            if frame_idx < 0:
                continue
            frame_tracks[frame_idx].append((int(tid), entry))

    track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    # Retrieve FPS from source video if available for video export
    fps = 30.0
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps_read = cap.get(cv2.CAP_PROP_FPS)
            if fps_read and fps_read > 1e-3:
                fps = float(fps_read)
        cap.release()

    video_out = os.path.join(vis_root, "overview.mp4")
    writer = None
    frame_size = None

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if imgfiles:
            sample = cv2.imread(imgfiles[0])
            if sample is not None:
                frame_size = (sample.shape[1], sample.shape[0])
                writer = cv2.VideoWriter(video_out, fourcc, max(1.0, fps / stride), frame_size)
                if not writer.isOpened():
                    writer.release()
                    writer = None
    except Exception:
        writer = None

    union_mask_cache: Dict[int, np.ndarray] = {}

    for frame_idx, img_path in enumerate(imgfiles):
        track_entries = frame_tracks.get(frame_idx, [])

        # Update track history regardless of sampling stride
        for tid, entry in track_entries:
            box_coords, _ = _extract_track_box(entry)
            if box_coords is None:
                continue
            x1, y1, x2, y2 = box_coords
            center = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
            history = track_history[tid]
            if not history or history[-1] != center:
                history.append(center)
                if len(history) > track_tail:
                    del history[: -track_tail]

        if frame_idx % stride != 0:
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        draw = frame.copy()

        # Overlay union mask
        if frame_idx not in union_mask_cache and frame_idx < len(masks):
            try:
                union_mask_cache[frame_idx] = masktool.decode(masks[frame_idx])
            except Exception:
                union_mask_cache[frame_idx] = None

        union_mask = union_mask_cache.get(frame_idx)
        if union_mask is not None:
            mask_arr = union_mask
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            if np.any(mask_arr):
                mask_overlay = np.zeros_like(draw)
                mask_overlay[mask_arr.astype(bool)] = (128, 128, 255)
                draw = cv2.addWeighted(draw, 0.65, mask_overlay, 0.35, 0.0)

        # Draw raw detection boxes
        if frame_idx < len(boxes):
            frame_boxes = boxes[frame_idx]
            if frame_boxes is not None:
                box_arr = np.array(frame_boxes)
                if box_arr.size > 0:
                    if box_arr.ndim == 1:
                        box_arr = box_arr.reshape(1, -1)
                    for det in box_arr:
                        x1, y1, x2, y2 = det[:4]
                        color = (0, 215, 255)
                        cv2.rectangle(draw,
                                      (int(round(x1)), int(round(y1))),
                                      (int(round(x2)), int(round(y2))),
                                      color,
                                      1)
                        if det.shape[0] > 4:
                            score = float(det[4])
                            cv2.putText(draw,
                                        f"{score:.2f}",
                                        (int(round(x1)), int(round(y1)) + 14),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45,
                                        color,
                                        1,
                                        cv2.LINE_AA)

        # Draw track-specific overlays
        for tid, entry in track_entries:
            box_coords, score = _extract_track_box(entry)
            if box_coords is None:
                continue
            x1, y1, x2, y2 = box_coords
            color = _color_from_track_id(tid)

            # Track mask contour
            track_mask = None
            if entry.get("rle") is not None:
                try:
                    track_mask = masktool.decode(entry["rle"])
                except Exception:
                    track_mask = None

            if track_mask is not None:
                mask_img = track_mask
                if mask_img.ndim == 3:
                    mask_img = mask_img[..., 0]
                mask_img = mask_img.astype(np.uint8)
                if np.any(mask_img):
                    contours_info = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
                    cv2.drawContours(draw, contours, -1, color, 2)

            cv2.rectangle(draw,
                          (int(round(x1)), int(round(y1))),
                          (int(round(x2)), int(round(y2))),
                          color,
                          2)

            label = f"id {tid}"
            if entry.get("det") and score is not None:
                label += f" | {score:.2f}"
            cv2.putText(draw,
                        label,
                        (int(round(x1)), max(12, int(round(y1)) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA)

            history = track_history.get(tid, [])
            if history:
                pts = np.array([[int(round(x)), int(round(y))] for x, y in history], dtype=np.int32)
                if pts.shape[0] >= 2:
                    cv2.polylines(draw, [pts.reshape(-1, 1, 2)], False, color, 2)
                cv2.circle(draw, tuple(pts[-1]), 4, color, -1)

        out_path = os.path.join(frame_dir, f"{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, draw)
        if writer is not None:
            if frame_size is None:
                frame_size = (draw.shape[1], draw.shape[0])
            writer.write(draw)

    if writer is not None:
        writer.release()

    # Camera trajectory plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for projection registration
    except Exception:
        print("Matplotlib not available; skipped camera trajectory visualization.")
    else:
        cam_trans = camera.get("world_cam_T")
        if cam_trans is not None:
            cam_trans = np.asarray(cam_trans)
        if cam_trans is not None and cam_trans.ndim == 2 and cam_trans.shape[1] == 3:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(cam_trans[:, 0], cam_trans[:, 1], cam_trans[:, 2], color="royalblue", linewidth=2)
            ax.scatter(cam_trans[0, 0], cam_trans[0, 1], cam_trans[0, 2], color="green", label="start")
            ax.scatter(cam_trans[-1, 0], cam_trans[-1, 1], cam_trans[-1, 2], color="red", label="end")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(vis_root, "camera_path_3d.png"), dpi=200)
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.plot(cam_trans[:, 0], cam_trans[:, 2], marker="o", markersize=2, linewidth=1.5)
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            ax2.set_title("Camera Top-Down Trajectory")
            ax2.grid(True, linestyle="--", alpha=0.4)
            fig2.tight_layout()
            fig2.savefig(os.path.join(vis_root, "camera_path_topdown.png"), dpi=200)
            plt.close(fig2)


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="input video")
parser.add_argument("--output_dir", type=str, default=None,
                    help="output directory (default: results/<video-stem>)")
parser.add_argument("--static_camera", action='store_true', help='whether the camera is static')
parser.add_argument("--visualize_mask", action='store_true', help='save deva vos for visualization')
parser.add_argument("--slam_backend", type=str, choices=get_available_backends(), default="droid",
                    help="camera backend (default: masked DROID-SLAM)")
parser.add_argument("--focal", type=float, default=None,
                    help="fixed focal length in pixels; skips the noisy focal search")
parser.add_argument("--visualize", action='store_true',
                    help='generate overlay visualizations for masks, boxes, tracks, and camera path')
parser.add_argument("--visualize_stride", type=int, default=1,
                    help='frame sampling stride when dumping visualization frames (default: 1)')
args = parser.parse_args()
if args.focal is not None and (not np.isfinite(args.focal) or args.focal <= 0):
    parser.error("--focal must be a finite positive value")

file = args.video
seq = os.path.basename(file).split('.')[0]

seq_folder = args.output_dir or f"results/{seq}"
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)

print('Extracting frames ...')
nframes = video2frames(file, img_folder)

print('Detect, Segment, and Track ...')
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25,
                                               min_size=100, save_vos=args.visualize_mask)
default_backend = os.environ.get("TROPHIES_SLAM_BACKEND", "droid")
print(f"Running {args.slam_backend or default_backend} backend ...")
masks = np.array([masktool.decode(m) for m in masks_])
masks = torch.from_numpy(masks)

cam_int, is_static = calibrate_intrinsics(
    img_folder,
    masks,
    backend=args.slam_backend,
    is_static=args.static_camera,
    focal=args.focal,
)
cam_R, cam_T = run_metric_slam(
    img_folder,
    masks=masks,
    calib=cam_int,
    is_static=is_static,
    backend=args.slam_backend,
)
wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(),
          'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
          'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}

np.save(f'{seq_folder}/camera.npy', camera)
np.save(f'{seq_folder}/boxes.npy', boxes_)
np.save(f'{seq_folder}/masks.npy', masks_)
np.save(f'{seq_folder}/tracks.npy', tracks_)

if args.visualize:
    print('Generating visualizations ...')
    visualize_estimate_results(
        seq_folder,
        video_path=file,
        imgfiles=imgfiles,
        boxes=boxes_,
        masks=masks_,
        tracks=tracks_,
        camera=camera,
        stride=args.visualize_stride,
    )
