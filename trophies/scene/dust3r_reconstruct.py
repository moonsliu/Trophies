"""Run DUSt3R scene reconstruction behind a Trophies configuration layer."""

from __future__ import annotations

import importlib
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tyro
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Options:
    config: Path = Path("configs/dust3r.yaml")
    """YAML config with DUSt3R, input, and output settings."""

    img_dir: Optional[Path] = None
    """Override input.image_dir."""

    output_dir: Optional[Path] = None
    """Override output.dir."""

    model_path: Optional[Path] = None
    """Override dust3r.model_path."""

    device: Optional[str] = None
    """Override dust3r.device. Use 'auto' to choose cuda when available."""

    human_masks: Optional[Path] = None
    """Optional COCO-RLE masks.npy enabling human-aware temporal attention."""

    video: Optional[Path] = None
    """Source video used to infer sampled frame indices when human_masks is set."""

    frame_indices: Optional[str] = None
    """Comma-separated source-video frame indices; overrides inference from video."""


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _config_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return section


def _maybe_add_dust3r_repo_to_path() -> None:
    repo_path = Path(__file__).resolve().parents[2] / "third_party" / "dust3r"
    for candidate in (repo_path, repo_path.parent):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _import_dust3r_modules() -> Dict[str, Any]:
    _maybe_add_dust3r_repo_to_path()
    prefixes = ("dust3r.dust3r", "dust3r")
    last_error: Optional[BaseException] = None

    for prefix in prefixes:
        try:
            return {
                "model": importlib.import_module(f"{prefix}.model"),
                "inference": importlib.import_module(f"{prefix}.inference"),
                "image_pairs": importlib.import_module(f"{prefix}.image_pairs"),
                "image": importlib.import_module(f"{prefix}.utils.image"),
                "device": importlib.import_module(f"{prefix}.utils.device"),
                "cloud_opt": importlib.import_module(f"{prefix}.cloud_opt"),
            }
        except ModuleNotFoundError as exc:
            last_error = exc

    raise ModuleNotFoundError(
        "Could not import the pinned DUSt3R submodule. Run: git submodule update --init --recursive"
    ) from last_error


def _select_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _collect_images(img_dir: Path, pattern: str) -> List[Path]:
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Expected an image directory: {img_dir}")

    paths = [
        path
        for path in sorted(img_dir.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise FileNotFoundError(f"No images matching '{pattern}' found in {img_dir}")
    return paths


def _prepare_dynamic_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    """Apply DUSt3R's resize and center-crop geometry to a binary mask."""

    from PIL import Image

    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    width1, height1 = image.size
    if image_size == 224:
        long_edge = round(image_size * max(width1 / height1, height1 / width1))
    else:
        long_edge = image_size
    scale = long_edge / float(max(image.size))
    new_size = tuple(int(round(value * scale)) for value in image.size)
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize(new_size, resampling.NEAREST)

    width, height = image.size
    center_x, center_y = width // 2, height // 2
    if image_size == 224:
        half = min(center_x, center_y)
        crop = (center_x - half, center_y - half, center_x + half, center_y + half)
    else:
        half_width = ((2 * center_x) // 16) * 8
        half_height = ((2 * center_y) // 16) * 8
        if width == height:
            half_height = 3 * half_width / 4
        crop = (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )
    return np.asarray(image.crop(crop), dtype=np.uint8) > 0


def _load_images_with_dynamic_masks(
    modules: Dict[str, Any],
    image_paths: List[Path],
    image_size: int,
    masks_path: Path,
    video: Optional[Path],
    frame_indices_value: Optional[str],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Load RGB inputs and attach aligned human masks/timestamps for the fork."""

    import torch

    from trophies.scene.dynamic_masks import decode_rle_mask, parse_frame_indices

    if not masks_path.is_file():
        raise FileNotFoundError(f"Human masks not found: {masks_path}")
    images = modules["image"].load_images(
        [str(path) for path in image_paths], size=image_size, verbose=True
    )
    frame_indices = parse_frame_indices(frame_indices_value, video, len(images))
    masks = np.load(masks_path, allow_pickle=True)
    if frame_indices.max(initial=0) >= len(masks):
        raise ValueError("frame indices reference entries outside masks.npy")

    for image, frame_idx in zip(images, frame_indices.tolist()):
        dynamic_mask = _prepare_dynamic_mask(decode_rle_mask(masks[frame_idx]), image_size)
        expected_shape = tuple(int(value) for value in image["img"].shape[-2:])
        if dynamic_mask.shape != expected_shape:
            raise ValueError(
                f"Dynamic mask shape {dynamic_mask.shape} does not match DUSt3R image {expected_shape}"
            )
        image["dynamic_mask"] = torch.from_numpy(dynamic_mask)[None, None]
        image["timestamp"] = torch.tensor([frame_idx], dtype=torch.int64)
    return images, frame_indices


def _run_dust3r(
    modules: Dict[str, Any],
    image_paths: List[Path],
    model_path: Path,
    device: str,
    image_size: int,
    schedule: str,
    niter: int,
    scenegraph_type: str,
    winsize: int,
    refid: int,
    loaded_images: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"DUSt3R checkpoint not found: {model_path}")

    model_cls = modules["model"].AsymmetricCroCo3DStereo
    inference = modules["inference"].inference
    make_pairs = modules["image_pairs"].make_pairs
    load_images = modules["image"].load_images
    to_numpy = modules["device"].to_numpy
    global_aligner = modules["cloud_opt"].global_aligner
    aligner_mode = modules["cloud_opt"].GlobalAlignerMode

    # DUSt3R checkpoints include argparse.Namespace metadata. PyTorch 2.6+
    # keeps weights_only=True by default, so allowlist only that metadata class.
    import argparse
    import torch.serialization

    torch.serialization.add_safe_globals([argparse.Namespace])
    model = model_cls.from_pretrained(str(model_path)).to(device)
    imgs = loaded_images or load_images([str(path) for path in image_paths], size=image_size, verbose=True)
    if len(imgs) == 1:
        import copy

        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    if scenegraph_type == "swin":
        scenegraph = f"{scenegraph_type}-{winsize}"
    elif scenegraph_type == "oneref":
        scenegraph = f"{scenegraph_type}-{refid}"
    else:
        scenegraph = scenegraph_type

    pairs = make_pairs(imgs, scene_graph=scenegraph, prefilter=None, symmetrize=True)
    network_output = inference(pairs, model, device, batch_size=1, verbose=True)

    mode = aligner_mode.PointCloudOptimizer if len(imgs) > 2 else aligner_mode.PairViewer
    scene = global_aligner(network_output, device=device, mode=mode, verbose=True)
    if mode == aligner_mode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=0.01)
        print(f"final loss: {loss}")

    result = {
        "rgbimg": scene.imgs,
        "intrinsics": to_numpy(scene.get_intrinsics()),
        "cam2world": to_numpy(scene.get_im_poses()),
        "pts3d": to_numpy(scene.get_pts3d()),
        "depths": to_numpy(scene.get_depthmaps()),
        "msk": to_numpy(scene.get_masks()),
        "conf": to_numpy([conf for conf in scene.im_conf]),
    }
    return result, network_output


def _build_payload(
    image_paths: List[Path],
    dust3r_result: Dict[str, Any],
    settings: Dict[str, Any],
    save_network_output: bool,
    network_output: Any,
) -> Dict[str, Any]:
    env: Dict[str, Dict[str, Any]] = {}
    for idx, image_path in enumerate(image_paths):
        key = image_path.stem
        env[key] = {
            "rgbimg": dust3r_result["rgbimg"][idx],
            "intrinsic": dust3r_result["intrinsics"][idx],
            "cam2world": dust3r_result["cam2world"][idx],
            "pts3d": dust3r_result["pts3d"][idx],
            "depths": dust3r_result["depths"][idx],
            "msk": dust3r_result["msk"][idx],
            "conf": dust3r_result["conf"][idx],
            "source_path": str(image_path),
        }

    payload: Dict[str, Any] = {
        "source_images": [str(path) for path in image_paths],
        "dust3r_ga_output": env,
        "settings": settings,
    }
    if save_network_output:
        payload["dust3r_network_output"] = network_output
    return payload


def run(opts: Options) -> Path:
    config = _read_yaml(opts.config)
    dust3r_cfg = _config_section(config, "dust3r")
    input_cfg = _config_section(config, "input")
    output_cfg = _config_section(config, "output")

    img_dir = opts.img_dir or Path(input_cfg.get("image_dir", "demo_data/quick_demo/frames"))
    output_dir = opts.output_dir or Path(output_cfg.get("dir", "outputs/dust3r_demo"))
    model_path = opts.model_path or Path(dust3r_cfg.get("model_path", "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"))
    device = _select_device(opts.device or str(dust3r_cfg.get("device", "auto")))

    image_paths = _collect_images(img_dir, str(input_cfg.get("glob", "*")))
    modules = _import_dust3r_modules()
    loaded_images = None
    frame_indices = None
    if opts.human_masks is not None:
        loaded_images, frame_indices = _load_images_with_dynamic_masks(
            modules,
            image_paths,
            int(dust3r_cfg.get("image_size", 512)),
            opts.human_masks,
            opts.video,
            opts.frame_indices,
        )

    settings = {
        "model_path": str(model_path),
        "image_size": int(dust3r_cfg.get("image_size", 512)),
        "schedule": str(dust3r_cfg.get("schedule", "linear")),
        "niter": int(dust3r_cfg.get("niter", 300)),
        "scenegraph_type": str(dust3r_cfg.get("scenegraph_type", "complete")),
        "winsize": int(dust3r_cfg.get("winsize", 1)),
        "refid": int(dust3r_cfg.get("refid", 0)),
        "device": device,
        "human_aware_attention": loaded_images is not None,
        "human_masks": str(opts.human_masks) if opts.human_masks is not None else None,
        "frame_indices": frame_indices.tolist() if frame_indices is not None else None,
    }

    result, network_output = _run_dust3r(
        modules=modules,
        image_paths=image_paths,
        model_path=model_path,
        device=device,
        image_size=settings["image_size"],
        schedule=settings["schedule"],
        niter=settings["niter"],
        scenegraph_type=settings["scenegraph_type"],
        winsize=settings["winsize"],
        refid=settings["refid"],
        loaded_images=loaded_images,
    )

    payload = _build_payload(
        image_paths=image_paths,
        dust3r_result=result,
        settings=settings,
        save_network_output=bool(dust3r_cfg.get("save_network_output", False)),
        network_output=network_output,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_name = str(output_cfg.get("scene_name", "scene"))
    output_path = output_dir / f"{scene_name}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Scene reconstruction written to {output_path}")
    return output_path


def main() -> None:
    run(tyro.cli(Options))


if __name__ == "__main__":
    main()
