"""Utility samplers used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class StandardNormalSampler:
    """Sample from a standard normal distribution."""

    dimension: int

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        return torch.randn(batch_size, self.dimension, device=device, dtype=dtype)


class UniformTimeSampler:
    """Sample times uniformly from the unit interval."""

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        return torch.rand(batch_size, 1, device=device, dtype=dtype)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""

    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
