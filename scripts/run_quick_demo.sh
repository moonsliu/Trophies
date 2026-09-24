#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO="${VIDEO_PATH:-}"
WORK_DIR="demo_data/quick_demo"
RAW_OUTPUT_DIR="outputs/dust3r_single"
ALIGNED_OUTPUT_DIR="outputs/dust3r_single_sim3"
HUMAN_OUTPUT_DIR="outputs/single_human_scene"
NUM_FRAMES=8
SLAM_BACKEND="droid"
FOCAL=""
SKIP_CAMERA=0
SKIP_HUMANS=0
SKIP_FRAMES=0
SKIP_DUST3R=0
SKIP_ALIGNMENT=0
SKIP_HUMAN_OPTIMIZATION=0
START_VIEWER=0
HOST="127.0.0.1"
PORT=8080
CONFIDENCE_THRESHOLD=1.5

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_quick_demo.sh --video /path/to/video.mp4 [options]

Pipeline:
  video -> SLAM trajectory/masks -> temporal SMPL -> free-camera DUSt3R
        -> Sim(3) alignment -> optimization

Options:
  --video PATH                 Input moving-camera video (or set VIDEO_PATH).
  --work-dir PATH              Demo inputs/intermediates (default: demo_data/quick_demo).
  --raw-output-dir PATH        Raw DUSt3R output (default: outputs/dust3r_single).
  --aligned-output-dir PATH    Sim(3)-aligned output (default: outputs/dust3r_single_sim3).
  --human-output-dir PATH      Optimized SMPL output (default: outputs/single_human_scene).
  --num-frames N               Uniformly sampled DUSt3R frames (default: 8).
  --slam-backend NAME          Camera backend (default: droid).
  --focal PIXELS               Fixed camera focal length; skips focal search.
  --skip-camera                Reuse camera.npy and masks.npy.
  --skip-humans                Reuse slam/hps/hps_track_*.npy.
  --skip-frames                Reuse sampled images and frame_indices.json.
  --skip-dust3r                Reuse raw DUSt3R scene.pkl.
  --skip-alignment             Reuse aligned scene.pkl.
  --skip-human-optimization    Reuse optimized SMPL output.
  --view                       Start Viser after reconstruction.
  --host HOST                  Viser host (default: 127.0.0.1).
  --port PORT                  Exact Viser port; must be free (default: 8080).
  --confidence-threshold FLOAT Viewer threshold for relaxed geometry (default: 1.5).
  -h, --help                   Show this help.

Examples:
  bash scripts/run_quick_demo.sh --video /path/to/video.mp4
  bash scripts/run_quick_demo.sh --video /path/to/video.mp4 --view
  bash scripts/run_quick_demo.sh --video /path/to/video.mp4 \
    --skip-camera --skip-humans --skip-frames --skip-dust3r \
    --skip-alignment --skip-human-optimization --view
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --raw-output-dir) RAW_OUTPUT_DIR="$2"; shift 2 ;;
    --aligned-output-dir) ALIGNED_OUTPUT_DIR="$2"; shift 2 ;;
    --human-output-dir) HUMAN_OUTPUT_DIR="$2"; shift 2 ;;
    --num-frames) NUM_FRAMES="$2"; shift 2 ;;
    --slam-backend) SLAM_BACKEND="$2"; shift 2 ;;
    --focal) FOCAL="$2"; shift 2 ;;
    --skip-camera) SKIP_CAMERA=1; shift ;;
    --skip-humans) SKIP_HUMANS=1; shift ;;
    --skip-frames) SKIP_FRAMES=1; shift ;;
    --skip-dust3r) SKIP_DUST3R=1; shift ;;
    --skip-alignment) SKIP_ALIGNMENT=1; shift ;;
    --skip-human-optimization) SKIP_HUMAN_OPTIMIZATION=1; shift ;;
    --view) START_VIEWER=1; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --confidence-threshold) CONFIDENCE_THRESHOLD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  echo "Missing --video PATH (or VIDEO_PATH)." >&2
  usage >&2
  exit 2
fi
if [[ ! "$NUM_FRAMES" =~ ^[0-9]+$ ]] || (( NUM_FRAMES < 3 )); then
  echo "--num-frames must be an integer of at least 3 for Sim(3) alignment." >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "--port must be an integer between 1 and 65535." >&2
  exit 2
fi

resolve_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "$1"
  fi
}

check_viewer_port() {
  python - "$HOST" "$PORT" <<'PYPORT'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
address = host if family == socket.AF_INET6 else socket.gethostbyname(host)
sock = socket.socket(family, socket.SOCK_STREAM)
try:
    sock.bind((address, port))
except OSError as exc:
    print(f"Viewer port {host}:{port} is unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PYPORT
}

VIDEO="$(resolve_path "$VIDEO")"
WORK_DIR="$(resolve_path "$WORK_DIR")"
RAW_OUTPUT_DIR="$(resolve_path "$RAW_OUTPUT_DIR")"
ALIGNED_OUTPUT_DIR="$(resolve_path "$ALIGNED_OUTPUT_DIR")"
HUMAN_OUTPUT_DIR="$(resolve_path "$HUMAN_OUTPUT_DIR")"
SLAM_DIR="${WORK_DIR}/slam"
FRAMES_DIR="${WORK_DIR}/frames"
FRAME_MANIFEST="${FRAMES_DIR}/frame_indices.json"
RAW_SCENE="${RAW_OUTPUT_DIR}/scene.pkl"
ALIGNED_SCENE="${ALIGNED_OUTPUT_DIR}/scene.pkl"
CAMERA_FILE="${SLAM_DIR}/camera.npy"
MASK_FILE="${SLAM_DIR}/masks.npy"

if [[ ! -f "$VIDEO" ]]; then
  echo "Video not found: $VIDEO" >&2
  exit 1
fi

mkdir -p "$WORK_DIR" "$SLAM_DIR" "$FRAMES_DIR" "$RAW_OUTPUT_DIR" "$ALIGNED_OUTPUT_DIR" "$HUMAN_OUTPUT_DIR"
cd "$PROJECT_ROOT"

run_stage() {
  local stage="$1"
  shift
  printf '\n[%s]\n' "$stage"
  printf '  %q' "$@"
  printf '\n'
  "$@"
}

if (( SKIP_CAMERA == 0 )); then
  CAMERA_COMMAND=(
    python scripts/estimate_camera.py
    --video "$VIDEO"
    --output_dir "$SLAM_DIR"
    --slam_backend "$SLAM_BACKEND"
  )
  if [[ -n "$FOCAL" ]]; then
    CAMERA_COMMAND+=(--focal "$FOCAL")
  fi
  run_stage "1/6 SLAM trajectory and human masks" "${CAMERA_COMMAND[@]}"
fi
if [[ ! -f "$CAMERA_FILE" || ! -f "$MASK_FILE" ]]; then
  echo "Camera stage requires $CAMERA_FILE and $MASK_FILE" >&2
  exit 1
fi

if (( SKIP_HUMANS == 0 )); then
  run_stage "2/6 Temporal SMPL reconstruction" \
    python scripts/estimate_humans.py \
      --input_dir "$SLAM_DIR" \
      --output_dir "$SLAM_DIR"
fi
if ! find "$SLAM_DIR/hps" -maxdepth 1 -type f -name 'hps_track_*.npy' -print -quit 2>/dev/null | grep -q .; then
  echo "Human stage requires $SLAM_DIR/hps/hps_track_*.npy" >&2
  exit 1
fi

if (( SKIP_FRAMES == 0 )); then
  run_stage "3/6 Sample DUSt3R frames" \
    python scripts/extract_video_frames.py "$VIDEO" \
      --output-dir "$FRAMES_DIR" \
      --num-frames "$NUM_FRAMES" \
      --clean-output-dir
fi
mapfile -t FRAME_FILES < <(find "$FRAMES_DIR" -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | sort)
if (( ${#FRAME_FILES[@]} < 3 )); then
  echo "At least three sampled images are required in $FRAMES_DIR" >&2
  exit 1
fi
if [[ ! -f "$FRAME_MANIFEST" ]]; then
  echo "Frame-index manifest not found: $FRAME_MANIFEST" >&2
  echo "Run without --skip-frames to regenerate sampled inputs." >&2
  exit 1
fi
FRAME_INDICES="$(python -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1]))["source_frame_indices"])))' "$FRAME_MANIFEST")"
IFS="," read -r -a FRAME_INDEX_ARRAY <<< "$FRAME_INDICES"
if (( ${#FRAME_FILES[@]} != ${#FRAME_INDEX_ARRAY[@]} )); then
  echo "Sampled image count does not match $FRAME_MANIFEST" >&2
  echo "Run without --skip-frames to regenerate a consistent set." >&2
  exit 1
fi

if (( SKIP_DUST3R == 0 )); then
  run_stage "4/6 Free-camera DUSt3R reconstruction" \
    python -m trophies.scene.dust3r_reconstruct \
      --img-dir "$FRAMES_DIR" \
      --human-masks "$MASK_FILE" \
      --frame-indices "$FRAME_INDICES" \
      --output-dir "$RAW_OUTPUT_DIR"
fi
if [[ ! -f "$RAW_SCENE" ]]; then
  echo "Raw DUSt3R scene not found: $RAW_SCENE" >&2
  exit 1
fi

if (( SKIP_ALIGNMENT == 0 )); then
  run_stage "5/6 DUSt3R-to-SLAM Sim(3) alignment" \
    python -m trophies.scene.dust3r_sim3_align \
      --scene-pkl "$RAW_SCENE" \
      --camera "$CAMERA_FILE" \
      --frame-indices "$FRAME_INDICES" \
      --human-masks "$MASK_FILE" \
      --output-dir "$ALIGNED_OUTPUT_DIR"
fi
if [[ ! -f "$ALIGNED_SCENE" ]]; then
  echo "Aligned scene not found: $ALIGNED_SCENE" >&2
  exit 1
fi

if (( SKIP_HUMAN_OPTIMIZATION == 0 )); then
  run_stage "6/6 Optimization" \
    python scripts/optimize_human_scene.py \
      --scene-pkl "$ALIGNED_SCENE" \
      --human-motion-dir "$SLAM_DIR" \
      --output-dir "$HUMAN_OUTPUT_DIR" \
      --smpl-model-dir "$PROJECT_ROOT/body_models/smpl"
fi
if ! find "$HUMAN_OUTPUT_DIR/hps" -maxdepth 1 -type f -name 'hps_track_*.npy' -print -quit 2>/dev/null | grep -q .; then
  echo "Optimization requires $HUMAN_OUTPUT_DIR/hps/hps_track_*.npy" >&2
  exit 1
fi

printf '\nQuick Commands completed.\n'
printf '  Raw DUSt3R scene: %s\n' "$RAW_SCENE"
printf '  Aligned scene:     %s\n' "$ALIGNED_SCENE"
printf '  SLAM trajectory:   %s\n' "$CAMERA_FILE"
printf '  SMPL parameters:   %s\n' "$HUMAN_OUTPUT_DIR/hps"
printf '  Source frame ids:  %s\n' "$FRAME_INDICES"

VIEW_COMMAND=(
  python -m trophies.vis.scene_viewer
  --scene-pkl "$ALIGNED_SCENE"
  --human-motion-dir "$HUMAN_OUTPUT_DIR"
  --smpl-model-dir "$PROJECT_ROOT/body_models/smpl"
  --show-smpl-meshes
  --display-frame world
  --no-use-scene-mask
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
  --host "$HOST"
  --port "$PORT"
)

if (( START_VIEWER == 1 )); then
  if ! check_viewer_port; then
    printf 'Choose another explicit port, for example: --port %d\n' "$((PORT + 1))" >&2
    printf 'If a previous demo owns the port, stop that Viewer before retrying.\n' >&2
    exit 1
  fi
  printf '\n[Viewer]\n'
  printf '  Binding exactly to %s:%s (automatic port switching is disabled).\n' "$HOST" "$PORT"
  printf '  On a remote server, forward remote port %s in VS Code or SSH.\n' "$PORT"
  exec "${VIEW_COMMAND[@]}"
fi

printf '\nVisualize with:\n '
printf ' %q' "${VIEW_COMMAND[@]}"
printf '\n'
