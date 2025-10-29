"""Data containers for training statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class TrainingHistory:
    steps: List[int] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, step: int, metrics: Dict[str, torch.Tensor]) -> None:
        self.steps.append(step)
        for key, value in metrics.items():
            self.metrics.setdefault(key, []).append(float(value.detach().cpu().item()))
