"""Extract evenly sampled frames from a video for DUSt3R demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for extracted frames.")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample.")
    parser.add_argument("--prefix", default="frame", help="Output filename prefix.")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Output image extension.")
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        help="Remove existing generated images with the selected prefix first.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Frame-index manifest path (default: OUTPUT_DIR/frame_indices.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    if not args.video.exists():
        raise FileNotFoundError(args.video)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video reports no frames: {args.video}")

    count = min(args.num_frames, total)
    last = max(total - 2, 0)
    requested_indices = (
        [0]
        if count == 1
        else [round(index * last / (count - 1)) for index in range(count)]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output_dir:
        for suffix in ("jpg", "jpeg", "png", "webp", "bmp"):
            for old_path in args.output_dir.glob(f"{args.prefix}_*.{suffix}"):
                old_path.unlink()

    written = []
    source_indices = []
    for output_index, requested_index in enumerate(requested_indices):
        frame = None
        actual_index = requested_index
        for candidate_index in range(requested_index, max(requested_index - 10, -1), -1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
            ok, candidate = capture.read()
            if ok:
                frame = candidate
                actual_index = candidate_index
                break
        if frame is None:
            raise RuntimeError(f"Failed to read frame {requested_index} from {args.video}")

        output_path = args.output_dir / f"{args.prefix}_{output_index:04d}.{args.ext}"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write {output_path}")
        written.append(output_path)
        source_indices.append(actual_index)

    capture.release()
    manifest_path = args.manifest or (args.output_dir / "frame_indices.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "video": str(args.video.resolve()),
        "total_video_frames": total,
        "source_frame_indices": source_indices,
        "images": [path.name for path in written],
    }
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Extracted {len(written)} frames from {args.video} to {args.output_dir}")
    print(f"Frame-index manifest: {manifest_path}")
    for path, source_index in zip(written, source_indices):
        print(f"{path} <- source frame {source_index}")


if __name__ == "__main__":
    main()
