"""Synthetic data utilities for Flow Matching experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


@dataclass
class SyntheticBatch:
    """Container bundling the tensors required for training."""

    x0: torch.Tensor
    x1: torch.Tensor
    t: torch.Tensor

    @property
    def xt(self) -> torch.Tensor:
        return (1.0 - self.t.unsqueeze(-1)) * self.x0 + self.t.unsqueeze(-1) * self.x1

    @property
    def target_velocity(self) -> torch.Tensor:
        return self.x1 - self.x0


class SyntheticPairDataset:
    """Base class for synthetic datasets defined by sampling functions."""

    def __init__(
        self,
        noise_sampler: Callable[[int, torch.device], torch.Tensor],
        data_sampler: Callable[[int, torch.device], torch.Tensor],
        device: torch.device,
    ) -> None:
        self._noise_sampler = noise_sampler
        self._data_sampler = data_sampler
        self.device = device

    def sample(self, batch_size: int) -> SyntheticBatch:
        x0 = self._noise_sampler(batch_size, self.device)
        x1 = self._data_sampler(batch_size, self.device)
        t = torch.rand(batch_size, device=self.device)
        return SyntheticBatch(x0=x0, x1=x1, t=t)


def build_1d_mixture_dataset(device: torch.device) -> SyntheticPairDataset:
    """Create a simple one-dimensional mixture-of-Gaussians dataset."""

    def sample_noise(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, 1, device=device)

    def sample_data(batch_size: int, device: torch.device) -> torch.Tensor:
        components = torch.randint(0, 2, (batch_size, 1), device=device)
        means = torch.tensor([-2.0, 2.0], device=device)
        stds = torch.tensor([0.3, 0.3], device=device)
        mean = means[components]
        std = stds[components]
        return mean + std * torch.randn(batch_size, 1, device=device)

    return SyntheticPairDataset(sample_noise, sample_data, device)


def build_2d_spiral_dataset(device: torch.device) -> SyntheticPairDataset:
    """Create a two-dimensional spiral dataset."""

    def sample_noise(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, 2, device=device)

    def sample_data(batch_size: int, device: torch.device) -> torch.Tensor:
        angles = torch.rand(batch_size, device=device) * 4.0 * torch.pi
        radii = torch.linspace(0.5, 2.0, batch_size, device=device)
        radii = radii[torch.randperm(batch_size, device=device)]
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        return torch.stack([x, y], dim=-1)

    return SyntheticPairDataset(sample_noise, sample_data, device)


class RectifiedPairDataset(SyntheticPairDataset):
    """Dataset containing pairs produced by rectified flow simulation."""

    def __init__(self, pairs: Dict[str, torch.Tensor], device: torch.device) -> None:
        self.pairs = {k: v.to(device) for k, v in pairs.items()}
        super().__init__(self._sample_noise, self._sample_data, device)

    def _sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:  # type: ignore[override]
        indices = torch.randint(0, self.pairs["x0"].shape[0], (batch_size,), device=device)
        return self.pairs["x0"][indices]

    def _sample_data(self, batch_size: int, device: torch.device) -> torch.Tensor:  # type: ignore[override]
        indices = torch.randint(0, self.pairs["x1"].shape[0], (batch_size,), device=device)
        return self.pairs["x1"][indices]

    def sample(self, batch_size: int) -> SyntheticBatch:  # type: ignore[override]
        indices = torch.randint(0, self.pairs["x0"].shape[0], (batch_size,), device=self.device)
        x0 = self.pairs["x0"][indices]
        x1 = self.pairs["x1"][indices]
        t = torch.rand(batch_size, device=self.device)
        return SyntheticBatch(x0=x0, x1=x1, t=t)
