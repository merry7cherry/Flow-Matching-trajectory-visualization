from __future__ import annotations

import math
from typing import Sequence

import torch

from .base import PairDataset, SampleBatch


class GaussianMixture1D(PairDataset):
    """1D mixture of Gaussians target distribution with a concentrated base."""

    def __init__(
        self,
        means: Sequence[float] = (-4.0, 4.0),
        std: float = 0.35,
        base_std: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=1, seed=seed)
        self.means = torch.tensor(means, dtype=torch.float32)
        self.std = std
        self.base_std = base_std

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(batch_size, 1, generator=self._generator)
        return (self.base_std * noise).to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        component_idx = torch.randint(
            low=0,
            high=len(self.means),
            size=(batch_size,),
            generator=self._generator,
        )
        means = self.means[component_idx].unsqueeze(1)
        noise = torch.randn(batch_size, 1, generator=self._generator)
        samples = means + self.std * noise
        return samples.to(device)


class GaussianMixture2D(PairDataset):
    """2D Gaussian mixture arranged on a square with separated modes."""

    def __init__(
        self,
        centers: Sequence[Sequence[float]] = ((-4.0, -4.0), (-4.0, 4.0), (4.0, -4.0), (4.5, 4.5)),
        std: float = 0.4,
        base_std: float = 0.6,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=2, seed=seed)
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.std = std
        self.base_std = base_std

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(batch_size, 2, generator=self._generator)
        return (self.base_std * noise).to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        component_idx = torch.randint(
            low=0,
            high=len(self.centers),
            size=(batch_size,),
            generator=self._generator,
        )
        centers = self.centers[component_idx]
        noise = torch.randn(batch_size, 2, generator=self._generator)
        samples = centers + self.std * noise
        return samples.to(device)


class EightGaussianToMoonDataset(PairDataset):
    """2D dataset mapping an outer ring of Gaussians to an inner two-moon shape."""

    def __init__(
        self,
        source_radius: float = 8.0,
        source_std: float = 0.35,
        target_radius: float = 2.5,
        target_horizontal_gap: float | None = None,
        target_vertical_gap: float = 1.5,
        target_std: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=2, seed=seed)
        angles = [2.0 * math.pi * i / 8 for i in range(8)]
        self.source_centers = torch.tensor(
            [
                (
                    source_radius * math.cos(theta),
                    source_radius * math.sin(theta),
                )
                for theta in angles
            ],
            dtype=torch.float32,
        )
        self.source_std = source_std
        self.target_radius = target_radius
        horizontal_gap = target_radius if target_horizontal_gap is None else target_horizontal_gap
        self.target_horizontal_gap = horizontal_gap
        self.target_vertical_gap = target_vertical_gap
        self.target_std = target_std
        self.upper_center = torch.tensor(
            (-horizontal_gap / 2.0, target_vertical_gap / 2.0), dtype=torch.float32
        )
        self.lower_center = torch.tensor(
            (horizontal_gap / 2.0, -target_vertical_gap / 2.0), dtype=torch.float32
        )

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        component_idx = torch.randint(
            low=0,
            high=self.source_centers.shape[0],
            size=(batch_size,),
            generator=self._generator,
        )
        centers = self.source_centers[component_idx]
        noise = torch.randn(batch_size, 2, generator=self._generator) * self.source_std
        samples = centers + noise
        return samples.to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        component_idx = torch.randint(
            low=0,
            high=2,
            size=(batch_size,),
            generator=self._generator,
        )
        theta = torch.rand(batch_size, generator=self._generator) * math.pi

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        arc = torch.stack((self.target_radius * cos_theta, self.target_radius * sin_theta), dim=1)
        upper_center = self.upper_center.to(device=arc.device, dtype=arc.dtype)
        lower_center = self.lower_center.to(device=arc.device, dtype=arc.dtype)
        upper_moon = arc + upper_center
        lower_moon = torch.stack(
            (self.target_radius * cos_theta, -self.target_radius * sin_theta),
            dim=1,
        ) + lower_center

        samples = torch.where(component_idx.unsqueeze(1) == 0, upper_moon, lower_moon)
        noise = torch.randn(batch_size, 2, generator=self._generator) * self.target_std
        samples = samples + noise
        return samples.to(device)


__all__ = [
    "GaussianMixture1D",
    "GaussianMixture2D",
    "EightGaussianToMoonDataset",
    "SampleBatch",
]
