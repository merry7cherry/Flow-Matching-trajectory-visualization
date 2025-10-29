"""Synthetic datasets used for Flow Matching trajectory visualisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SyntheticDataset:
    """Base class for synthetic datasets.

    The dataset is expected to expose a :meth:`sample` method returning batches
    of samples in the shape ``(batch_size, data_dim)``.
    """

    data_dim: int

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class GaussianMixture1D(SyntheticDataset):
    """A simple one dimensional Gaussian mixture dataset."""

    def __init__(
        self,
        means: tuple[float, float] = (-2.0, 2.0),
        stds: tuple[float, float] = (0.3, 0.3),
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        super().__init__(data_dim=1)
        if len(means) != len(stds) or len(means) != len(weights):
            raise ValueError("Means, standard deviations and weights must have the same length.")
        if not torch.isclose(torch.tensor(sum(weights)), torch.tensor(1.0)):
            raise ValueError("Mixture weights must sum to one.")
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.components = len(means)

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        categorical = torch.distributions.Categorical(self.weights)
        component_indices = categorical.sample((batch_size,))
        means = self.means.to(device=device, dtype=dtype)[component_indices]
        stds = self.stds.to(device=device, dtype=dtype)[component_indices]
        samples = torch.randn(batch_size, device=device, dtype=dtype) * stds + means
        return samples.unsqueeze(-1)


class TwoMoons2D(SyntheticDataset):
    """A light-weight implementation of the classical two moons dataset."""

    def __init__(
        self,
        radius: float = 2.0,
        width: float = 0.25,
        separation: float = 0.5,
    ) -> None:
        super().__init__(data_dim=2)
        self.radius = radius
        self.width = width
        self.separation = separation

    def _sample_single_moon(self, count: int, upper: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        angle = torch.rand(count, device=device, dtype=dtype) * torch.pi
        radius_noise = torch.randn(count, device=device, dtype=dtype) * self.width
        r = self.radius + radius_noise
        x = r * torch.cos(angle)
        y = r * torch.sin(angle)
        if upper:
            y = y + self.separation
        else:
            x = -x
            y = -y - self.separation
        return torch.stack([x, y], dim=-1)

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        bernoulli = torch.distributions.Bernoulli(probs=0.5)
        upper_mask = bernoulli.sample((batch_size,)).to(dtype=torch.bool)
        upper_count = int(upper_mask.sum().item())
        lower_count = batch_size - upper_count
        samples_upper = self._sample_single_moon(upper_count, True, device, dtype)
        samples_lower = self._sample_single_moon(lower_count, False, device, dtype)
        samples = torch.empty(batch_size, self.data_dim, device=device, dtype=dtype)
        if upper_count:
            samples[upper_mask] = samples_upper
        if lower_count:
            samples[~upper_mask] = samples_lower
        return samples
