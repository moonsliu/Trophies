"""Geometry operations needed by temporal human inference."""
import torch
from torch.nn import functional as F


def rotation_6d_to_matrix(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    first = F.normalize(values[:, :, 0], dim=-1)
    second_raw = values[:, :, 1]
    second = F.normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)
