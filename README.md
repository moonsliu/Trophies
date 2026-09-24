# Trophies

Official code organization for **Trophies: Temporal Reconstruction of Places, Humans, and Cameras from Multi-view Videos**.

Trophies reconstructs dynamic humans, static places, and cameras in a shared 4D world coordinate frame. This repository provides the public inference pipeline and reproducible Quick Commands workflow.

### Install

Initialize the pinned third-party forks and install the environment:

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate Trophies
bash scripts/setup_environment.sh
```

The installer uses the Blackwell-compatible PyTorch CUDA 12.8 stack by default:

- `torch==2.9.0+cu128`, `torchvision==0.24.0+cu128`, `torchaudio==2.9.0+cu128`;
- `numpy==1.26.4`, `opencv-python==4.10.0.84`, `pillow==10.4.0`, and `plyfile==1.0.3`;
- CroCo, the `moonsliu/dust3r` fork under `third_party/dust3r`, and the Trophies package;
- ViTDet, SAM, DEVA, masked DROID-SLAM, and the Blackwell-compatible `moonsliu/lietorch` fork;
- temporal human reconstruction, SMPL visualization, and PyTorch3D.

Override the PyTorch wheel target if needed:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  TORCH_VERSION=<torch-version> \
  TORCHVISION_VERSION=<torchvision-version> \
  TORCHAUDIO_VERSION=<torchaudio-version> \
  bash scripts/setup_environment.sh
```

The default CUDA 12.8 build is tested on Blackwell/sm_120 GPUs. Use the
override above when another CUDA-compatible PyTorch build is required.

### Environment Checks

```bash
python -c "import torch, numpy; print(torch.__version__, torch.version.cuda, numpy.__version__); print(torch.cuda.get_device_capability(0)); x=torch.randn(16,16,device='cuda'); print((x@x).shape)"
python scripts/estimate_camera.py --help
python scripts/estimate_humans.py --help
python -m trophies.scene.dust3r_reconstruct --help
python -m trophies.scene.dust3r_sim3_align --help
python scripts/optimize_human_scene.py --help
python -m trophies.vis.scene_viewer --help
python scripts/check_human_motion_env.py
bash scripts/run_quick_demo.sh --help
```

## Assets

The Quick Commands workflow needs the DUSt3R and camera-stage checkpoints, plus
the licensed neutral SMPL body model for mesh generation and optimization.

```text
checkpoints/
  DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth  # core
  droid.pth                                # core
  camcalib_sa_biased_l2.ckpt               # core
  DEVA-propagation.pth                     # core
  sam_vit_h_4b8939.pth                     # core
body_models/
  smpl/
    SMPL_NEUTRAL.pkl                       # mesh generation and optimization
```

### DUSt3R Checkpoint

Download the checkpoint listed by the pinned official DUSt3R submodule:

```bash
mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

### Camera Checkpoints

Download the four checkpoints required by `scripts/estimate_camera.py`:

```bash
mkdir -p checkpoints
wget -P checkpoints https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
gdown --fuzzy -O checkpoints/droid.pth https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing
gdown --fuzzy -O checkpoints/camcalib_sa_biased_l2.ckpt https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing
```

The ViTDet detector checkpoint is fetched from the official Detectron2 model
URL on first use. Moving-camera mode also loads ZoeDepth through `torch.hub` on
first use for metric-scale recovery. These downloads require network access.
Checkpoints remain subject to their upstream licenses and are not committed to
this repository.

### Native Camera Estimator

Camera estimation is implemented directly in this repository and does not
require a separate camera or motion checkout. The entry point is
`scripts/estimate_camera.py`, and the processing stages are:

1. Decode every video frame into the selected output directory.
2. Detect people with ViTDet, segment them with SAM, and maintain temporal
   identities with DEVA.
3. Pass the person masks into DROID-SLAM so human pixels do not contribute to
   camera tracking or bundle adjustment.
4. Recover metric translation scale from static background depth using
   ZoeDepth.
5. Estimate gravity orientation and rotate the trajectory into the Trophies
   world frame.

The implementation is organized under `trophies/pipeline/` for detection,
segmentation, and tracking, and `trophies/camera/` for masked SLAM, metric scale,
and world-frame alignment. The maintained
[`moonsliu/DROID-SLAM`](https://github.com/moonsliu/DROID-SLAM/tree/trophies)
fork is pinned under `third_party/droid_slam` as a Git submodule. It recursively
pins Eigen and the Blackwell-compatible
[`moonsliu/lietorch`](https://github.com/moonsliu/lietorch) fork.

### Body Model Assets

SMPL is license-restricted and is not redistributed with this repository. Follow
the same official download route documented by
[`hongsukchoi/HSfM_RELEASE`](https://github.com/hongsukchoi/HSfM_RELEASE#directory-structure):

1. Register, accept the model license, and download SMPL from the official
   [SMPL](https://smpl.is.tue.mpg.de/download.php) or
   [SMPLify](https://smplify.is.tue.mpg.de/download.php) website.
2. Extract the downloaded archive.
3. Place the neutral model at exactly:

   ```text
   body_models/smpl/SMPL_NEUTRAL.pkl
   ```

The viewer and optimization step instantiate the neutral SMPL model. Keep it local: body models,
checkpoints, private datasets, and generated outputs are intentionally ignored
by git and must not be committed.

## Quick Commands

The commands below reconstruct temporal SMPL motion and a moving-camera video with
free-camera DUSt3R, then aligns the DUSt3R cameras and geometry to the metric SLAM trajectory with a
global Sim(3). The SLAM trajectory is used only after DUSt3R reconstruction; it
is not fixed inside DUSt3R optimization.

Download the example video:

```bash
mkdir -p demo_data/quick_demo
gdown --fuzzy \
  -O demo_data/quick_demo/example_video.mp4 \
  "https://drive.google.com/file/d/1H6gyykajrk2JsBBxBIdt9Z49oKgYAuYJ/view?usp=share_link"
```

Then run the complete workflow:

```bash
conda activate Trophies
export VIDEO_PATH=demo_data/quick_demo/example_video.mp4
bash scripts/run_quick_demo.sh --video "$VIDEO_PATH" --focal 600 --view
```

To process another video, set `VIDEO_PATH` to its local path instead.

`--focal 600` reproduces the validated calibration of the reference clip. Omit
it for other videos to run automatic focal search, or provide calibrated
intrinsics when available.

The commands produce:

```text
demo_data/quick_demo/frames/frame_indices.json  # exact sampled source frames
demo_data/quick_demo/slam/camera.npy    # metric SLAM trajectory
demo_data/quick_demo/slam/masks.npy     # human masks
demo_data/quick_demo/slam/hps/          # raw temporal SMPL parameters
outputs/dust3r_single/scene.pkl              # raw DUSt3R frame
outputs/dust3r_single_sim3/scene.pkl         # SLAM world frame
outputs/single_human_scene/hps/              # optimized SMPL parameters
```

Human masks are used during DUSt3R temporal attention and again to remove people
from the exported static point cloud. The temporal human branch reconstructs
SMPL parameters, and optimization adjusts only the human translations
against the estimated scene floor before visualization.

## DUSt3R Fork

Trophies uses the [`main` branch of `moonsliu/dust3r`](https://github.com/moonsliu/dust3r/tree/main)
as a submodule pinned to an exact commit. This branch maintains human-aware
temporal cross-attention as regular source code while preserving the original
DUSt3R checkpoint structure.

Trophies-specific behavior remains in this repository whenever possible:

- reconstruction hyperparameters live in `configs/dust3r.yaml`;
- wrapper logic lives in `trophies/scene/dust3r_reconstruct.py`;
- fork-only changes should be limited to attention/masking internals that cannot be injected from the wrapper.

## Acknowledgements

This release builds on
[TRAM](https://github.com/yufu-wang/tram),
[HMR2.0 / 4D-Humans](https://github.com/shubham-goel/4D-Humans),
[DUSt3R](https://github.com/naver/dust3r),
[CroCo](https://github.com/naver/croco),
[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM),
[ZoeDepth](https://github.com/isl-org/ZoeDepth),
[Detectron2 and ViTDet](https://github.com/facebookresearch/detectron2),
[Segment Anything](https://github.com/facebookresearch/segment-anything),
[DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA),
[SMPL](https://smpl.is.tue.mpg.de/), [PyTorch3D](https://pytorch3d.org/), and
[Viser](https://github.com/nerfstudio-project/viser). We thank their authors
and contributors for releasing the code, models, and tools that make this
project possible.

## License

Original Trophies source code is released under the [MIT License](LICENSE).
Third-party components, model checkpoints, body models, and datasets remain
subject to their respective licenses. In particular, the complete pipeline
depends on components with non-commercial restrictions; the Trophies MIT
License does not override those terms. Refer to the license files in the pinned
upstream repositories before use or redistribution.

## Citation

```bibtex
@inproceedings{trophies:liu:2026,
title="{TROPHIES: Temporal Reconstruction of Places, Humans, and Cameras from Multi-view Videos}",
author={Jinpeng Liu and Yukang Xu and Yutong Li and Xingyu Liu},
booktitle={CVPR},
year={2026}
}
```
