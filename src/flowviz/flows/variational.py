from __future__ import annotations

import torch

from .base import FlowMatchingObjective


class VariationalFlowObjective(FlowMatchingObjective):
    """Variational flow matching keeps linear interpolation as baseline."""

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - t) * x0 + t * x1

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del xt, t
        return x1 - x0


__all__ = ["VariationalFlowObjective"]
