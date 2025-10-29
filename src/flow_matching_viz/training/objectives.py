"""Loss functions for Flow Matching."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlowMatchingObjective(nn.Module):
    """Default objective for standard Flow Matching."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.criterion(prediction, target)
