"""Image-track preprocessing for temporal human reconstruction."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


def _transform(point, center, scale, resolution, invert=False):
    height = 200.0 * scale + 1e-6
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0, 0] = float(resolution[1]) / height
    matrix[1, 1] = float(resolution[0]) / height
    matrix[0, 2] = resolution[1] * (-float(center[0]) / height + 0.5)
    matrix[1, 2] = resolution[0] * (-float(center[1]) / height + 0.5)
    matrix[2, 2] = 1.0
    if invert:
        matrix = np.linalg.inv(matrix)
    transformed = matrix @ np.array([point[0] - 1, point[1] - 1, 1.0])
    return transformed[:2].astype(int) + 1


def crop_person(image: np.ndarray, center: np.ndarray, scale: float, resolution=(256, 256)) -> np.ndarray:
    upper_left = _transform([1, 1], center, scale, resolution, invert=True) - 1
    bottom_right = _transform([resolution[0] + 1, resolution[1] + 1], center, scale, resolution, invert=True) - 1
    shape = [bottom_right[1] - upper_left[1], bottom_right[0] - upper_left[0]]
    if image.ndim > 2:
        shape.append(image.shape[2])
    cropped = np.zeros(shape)
    new_x = max(0, -upper_left[0]), min(bottom_right[0], image.shape[1]) - upper_left[0]
    new_y = max(0, -upper_left[1]), min(bottom_right[1], image.shape[0]) - upper_left[1]
    old_x = max(0, upper_left[0]), min(image.shape[1], bottom_right[0])
    old_y = max(0, upper_left[1]), min(image.shape[0], bottom_right[1])
    cropped[new_y[0]:new_y[1], new_x[0]:new_x[1]] = image[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    return resize(cropped, resolution).astype(np.uint8)


def split_contiguous_track(frames: np.ndarray, boxes: np.ndarray, min_length: int = 16):
    """Split detections at frame gaps and discard sequences shorter than the model window."""
    frames = np.asarray(frames)
    boxes = np.asarray(boxes)
    if len(frames) != len(boxes):
        raise ValueError("frames and boxes must have the same length")
    if len(frames) == 0:
        return [], []

    boundaries = np.flatnonzero(np.diff(frames) != 1) + 1
    indices = np.split(np.arange(len(frames)), boundaries)
    valid = [segment for segment in indices if len(segment) >= min_length]
    return [frames[segment] for segment in valid], [boxes[segment] for segment in valid]


def boxes_to_centers_scales(boxes: np.ndarray):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    width, height = x2 - x1, y2 - y1
    centers = np.stack([x1 + width / 2.0, y1 + height / 2.0], axis=1)
    return centers, np.maximum(width, height) / 200.0


class TrackDataset(Dataset):
    def __init__(self, image_files, boxes, image_focal=None, image_center=None, crop_size=256, dilation=1.2):
        self.image_files = image_files
        self.crop_size = crop_size
        self.centers, self.scales = boxes_to_centers_scales(boxes)
        self.scales *= dilation
        self.image_focal = image_focal
        self.image_center = image_center
        self.normalize = Compose([ToTensor(), Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = cv2.imread(str(self.image_files[index]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_files[index]}")
        image = image[:, :, ::-1]
        crop = crop_person(image, self.centers[index], self.scales[index], (self.crop_size, self.crop_size))
        height, width = image.shape[:2]
        focal = self.image_focal if self.image_focal is not None else np.sqrt(height ** 2 + width ** 2)
        center = self.image_center if self.image_center is not None else np.array([width / 2.0, height / 2.0])
        return {
            "img": self.normalize(crop),
            "img_idx": torch.tensor(index).long(),
            "scale": torch.tensor(self.scales[index]).float(),
            "center": torch.tensor(self.centers[index]).float(),
            "img_focal": torch.tensor(focal).float(),
            "img_center": torch.tensor(center).float(),
        }
