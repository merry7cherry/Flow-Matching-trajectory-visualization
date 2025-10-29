from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class SampleBatch:
    x0: torch.Tensor
    x1: torch.Tensor


class PairDataset(ABC):
    """Abstract base class for synthetic datasets that provide (x0, x1) pairs."""

    def __init__(self, dim: int, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    @abstractmethod
    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from the base (noise) distribution."""

    @abstractmethod
    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from the target data distribution."""

    def sample_pairs(self, batch_size: int, device: torch.device) -> SampleBatch:
        x0 = self.sample_base(batch_size, device)
        x1 = self.sample_target(batch_size, device)
        return SampleBatch(x0=x0, x1=x1)


class EmpiricalPairDataset(PairDataset):
    """Dataset backed by pre-computed (x0, x1) pairs."""

    def __init__(self, pairs: Tuple[torch.Tensor, torch.Tensor], seed: int = 42) -> None:
        x0, x1 = pairs
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must have matching shapes")
        if x0.ndim != 2:
            raise ValueError("Expected tensors of shape (N, dim)")
        super().__init__(dim=x0.shape[1], seed=seed)
        self._x0 = x0.detach().cpu()
        self._x1 = x1.detach().cpu()

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(
            low=0,
            high=self._x0.shape[0],
            size=(batch_size,),
            generator=self._generator,
        )
        return self._x0[idx].to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(
            low=0,
            high=self._x1.shape[0],
            size=(batch_size,),
            generator=self._generator,
        )
        return self._x1[idx].to(device)

    def sample_pairs(self, batch_size: int, device: torch.device) -> SampleBatch:
        idx = torch.randint(
            low=0,
            high=self._x0.shape[0],
            size=(batch_size,),
            generator=self._generator,
        )
        x0 = self._x0[idx].to(device)
        x1 = self._x1[idx].to(device)
        return SampleBatch(x0=x0, x1=x1)
