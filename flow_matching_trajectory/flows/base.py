"""Interfaces for Flow Matching variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import nn

from ..data.synthetic import SyntheticBatch


@dataclass
class FlowTrainingStep:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class FlowMatchingVariant(ABC):
    """Abstract base class capturing behaviour shared across variants."""

    def __init__(self, velocity_model: nn.Module, name: str) -> None:
        self.velocity_model = velocity_model
        self.name = name
        self.device = torch.device("cpu")

    @property
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.velocity_model.parameters()

    def to(self, device: torch.device) -> "FlowMatchingVariant":
        self.device = device
        self.velocity_model.to(device)
        for module in self._additional_modules():
            module.to(device)
        return self

    def _additional_modules(self) -> Iterable[nn.Module]:
        return []

    def train(self, mode: bool = True) -> "FlowMatchingVariant":  # type: ignore[override]
        self.velocity_model.train(mode)
        for module in self._additional_modules():
            module.train(mode)
        return self

    def eval(self) -> "FlowMatchingVariant":  # type: ignore[override]
        self.velocity_model.eval()
        for module in self._additional_modules():
            module.eval()
        return self

    @abstractmethod
    def compute_training_step(self, batch: SyntheticBatch) -> FlowTrainingStep:
        """Compute the loss and auxiliary metrics for a training step."""

    @abstractmethod
    def predict_velocity(
        self, xt: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict the velocity field for given states and time."""

    def sample_inference_context(
        self, batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Optional inference-time context (e.g. a latent code)."""

        return None
