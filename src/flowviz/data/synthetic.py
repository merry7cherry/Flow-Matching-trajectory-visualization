from __future__ import annotations

from typing import Sequence

import torch

from .base import PairDataset, SampleBatch


class GaussianMixture1D(PairDataset):
    """1D mixture of Gaussians target distribution with standard normal base."""

    def __init__(self, means: Sequence[float] = (-2.0, 2.5), std: float = 0.3, seed: int = 42) -> None:
        super().__init__(dim=1, seed=seed)
        self.means = torch.tensor(means, dtype=torch.float32)
        self.std = std

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(batch_size, 1, generator=self._generator)
        return noise.to(device)

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
    """2D Gaussian mixture arranged on a square."""

    def __init__(
        self,
        centers: Sequence[Sequence[float]] = ((-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.5, 2.5)),
        std: float = 0.35,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=2, seed=seed)
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.std = std

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(batch_size, 2, generator=self._generator)
        return noise.to(device)

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


__all__ = [
    "GaussianMixture1D",
    "GaussianMixture2D",
    "SampleBatch",
]
