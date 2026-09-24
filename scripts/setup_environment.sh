#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-Trophies}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.0}"
INSTALL_COMPILED="${INSTALL_COMPILED:-1}"
INSTALL_CAMERA_PYTHONPATH="${INSTALL_CAMERA_PYTHONPATH:-1}"
INSTALL_CONDA_DEPS="${INSTALL_CONDA_DEPS:-1}"
DETECTRON2_REF="${DETECTRON2_REF:-a59f05630a8f205756064244bf5beb8661f96180}"
PYTORCH3D_REF="${PYTORCH3D_REF:-stable}"
CROCO_REF="${CROCO_REF:-4969d91c4e141fa61ec6a0adef72008010c8756d}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_environment.sh

Installs the full Trophies environment:
  - DUSt3R scene reconstruction stack
  - camera, person tracking, and human-motion stack
  - compiled CUDA extensions needed by masked DROID-SLAM

Initialize the pinned third-party source dependencies before installation:
  git submodule update --init --recursive

Common environment variables:
  ENV_NAME              Conda environment name shown in messages. Default: Trophies
  PYTORCH_INDEX_URL     PyTorch wheel index. Default: https://download.pytorch.org/whl/cu128
  TORCH_VERSION         Default: 2.9.0
  TORCHVISION_VERSION   Default: 0.24.0
  TORCHAUDIO_VERSION    Default: 2.9.0
  CROCO_REF             hongsukchoi/croco revision used by the release
  INSTALL_COMPILED      Build compiled camera packages. Default: 1
  INSTALL_CONDA_DEPS    Install CUDA toolkit and SuiteSparse. Default: 1
  INSTALL_CAMERA_PYTHONPATH  Register local DROID and DEVA paths. Default: 1
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

require_active_conda() {
  if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Activate the ${ENV_NAME} conda environment first:" >&2
    echo "  conda env create -f environment.yml" >&2
    echo "  conda activate ${ENV_NAME}" >&2
    exit 1
  fi
}

install_base() {
  echo "Installing Trophies base environment into: ${CONDA_PREFIX}"

  python -m pip install --upgrade pip "setuptools<81" wheel

  # CUDA 12.8 is the tested default for Blackwell/sm_120 GPUs.
  python -m pip install \
    torch=="${TORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    --index-url "${PYTORCH_INDEX_URL}"

  python -m pip install --force-reinstall \
    numpy==1.26.4 \
    pillow==10.4.0 \
    MarkupSafe==2.1.5 \
    opencv-python==4.10.0.84 \
    plyfile==1.0.3

  python -m pip install --no-build-isolation git+https://github.com/mattloper/chumpy
  python -m pip install -r requirements/base.txt git+https://github.com/hongsukchoi/croco.git@${CROCO_REF}

  mkdir -p third_party checkpoints body_models

  if [ ! -f third_party/dust3r/dust3r/model.py ]; then
    echo "Missing the Trophies DUSt3R submodule." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
  fi

  python -m pip install -e .

  echo "Base environment ready. DUSt3R is loaded from ./third_party/dust3r."
}

install_human_motion() {
  echo "Installing camera and human-motion environment."

  if [ "${INSTALL_CONDA_DEPS}" = "1" ]; then
    conda install -y -c "nvidia/label/cuda-12.8.0" cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 cuda-libraries-dev=12.8
    conda install -y -c conda-forge suitesparse
  else
    echo "Skipping conda-level camera packages because INSTALL_CONDA_DEPS=0."
  fi

  export CUDA_HOME="${CONDA_PREFIX}"
  export CUDA_PATH="${CONDA_PREFIX}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  CUDA_TARGET_DIR="${CONDA_PREFIX}/targets/x86_64-linux"
  export CPATH="${CUDA_TARGET_DIR}/include:${CPATH:-}"
  export C_INCLUDE_PATH="${CUDA_TARGET_DIR}/include:${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="${CUDA_TARGET_DIR}/include:${CPLUS_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${CUDA_TARGET_DIR}/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_TARGET_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
  export MAX_JOBS="${MAX_JOBS:-8}"

  python -m pip install --upgrade pip "setuptools<81" wheel ninja
  python -m pip install -r requirements/human_motion.txt

  if [ "${INSTALL_COMPILED}" = "1" ]; then
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_REF}"
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"


    if [ ! -f "${PROJECT_ROOT}/third_party/droid_slam/setup.py" ]; then
      echo "Missing third_party/droid_slam. Run: git submodule update --init --recursive" >&2
      exit 1
    fi
    python -m pip install --no-build-isolation "${PROJECT_ROOT}/third_party/droid_slam/thirdparty/pytorch_scatter"

    (
      cd "${PROJECT_ROOT}/third_party/droid_slam"
      python setup.py install
    )
  else
    echo "Skipping compiled camera packages because INSTALL_COMPILED=0."
  fi

  if [ "${INSTALL_CAMERA_PYTHONPATH}" = "1" ]; then
    SITE_PACKAGES=$(python - <<'PYI'
import site
print(site.getsitepackages()[0])
PYI
)
    PTH_FILE="${SITE_PACKAGES}/trophies_camera_paths.pth"
    cat > "${PTH_FILE}" <<EOF
${PROJECT_ROOT}/third_party/droid_slam
${PROJECT_ROOT}/third_party/droid_slam/droid_slam
${PROJECT_ROOT}/third_party/deva

EOF
    echo "Registered camera-estimation Python paths in ${PTH_FILE}"
  fi

  if [ "${INSTALL_COMPILED}" = "1" ]; then
    python scripts/check_human_motion_env.py
  else
    echo "Skipping full human-motion dependency check because INSTALL_COMPILED=0."
  fi

  echo "Camera and human-motion environment ready."
}

require_active_conda
install_base
install_human_motion

echo "Full Trophies environment setup finished."
