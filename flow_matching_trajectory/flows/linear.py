"""Standard Flow Matching using linear interpolation targets."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..data.synthetic import SyntheticBatch
from ..models.mlp import TimeConditionedMLP
from .base import FlowMatchingVariant, FlowTrainingStep


class LinearFlowMatching(FlowMatchingVariant):
    """Canonical Flow Matching variant with linear interpolation targets."""

    def __init__(self, dim: int, hidden_dim: int = 128, hidden_layers: int = 3) -> None:
        velocity_model = TimeConditionedMLP(
            in_dim=dim,
            hidden_dims=[hidden_dim] * hidden_layers,
            out_dim=dim,
        )
        super().__init__(velocity_model, name="linear_flow_matching")

    def compute_training_step(self, batch: SyntheticBatch) -> FlowTrainingStep:
        pred = self.predict_velocity(batch.xt, batch.t)
        target = batch.target_velocity
        loss = F.mse_loss(pred, target)
        metrics = {"loss": loss.detach(), "mse": loss.detach()}
        return FlowTrainingStep(loss=loss, metrics=metrics)

    def predict_velocity(
        self, xt: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.velocity_model(xt, t)
