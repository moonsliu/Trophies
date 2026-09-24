"""Camera orientation regressor used for world-frame gravity alignment."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, ResNet


class ResNet50Features(ResNet):
    """Torchvision ResNet-50 that returns the final spatial feature map."""

    def __init__(self) -> None:
        super().__init__(Bottleneck, [3, 4, 6, 3])
        del self.avgpool
        del self.fc

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv1(images)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return self.layer4(features)


def _distribution_to_angle(logits: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    positions = torch.linspace(-1.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    normalized_index = (torch.softmax(logits, dim=-1) * positions).sum(dim=-1)
    return (maximum - minimum) * (normalized_index + 1.0) / 2.0 + minimum


class CameraRegressorNetwork(nn.Module):
    """Predict vertical field of view, pitch, and roll from one RGB image."""

    def __init__(self, num_out_channels: int = 256) -> None:
        super().__init__()
        self.backbone = ResNet50Features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_vfov = nn.Linear(2048, num_out_channels)
        self.fc_pitch = nn.Linear(2048, num_out_channels)
        self.fc_roll = nn.Linear(2048, num_out_channels)
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(600, max_size=1000),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def forward(self, images, transform_data: bool = True):
        if transform_data:
            if isinstance(images, torch.Tensor):
                images = images.detach().cpu().numpy()
            images = self.data_transform(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = images.to(self.fc_vfov.weight.device)
        features = self.avgpool(self.backbone(images)).flatten(1)
        vfov = _distribution_to_angle(self.fc_vfov(features), 0.2617, 2.1)
        pitch = _distribution_to_angle(self.fc_pitch(features), -0.6, 0.6)
        roll = _distribution_to_angle(self.fc_roll(features), -0.6, 0.6)
        to_numpy = lambda value: value.detach().cpu().numpy().squeeze()
        return [to_numpy(vfov), to_numpy(pitch), to_numpy(roll)]

    def load_ckpt(self, checkpoint: str | Path | Mapping[str, object]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if isinstance(checkpoint, (str, Path)) else checkpoint
        source = payload["state_dict"]
        state_dict = OrderedDict(
            (key.removeprefix("model."), value)
            for key, value in source.items()
            if key.startswith("model.")
        )
        self.load_state_dict(state_dict, strict=True)
        return self.eval()
