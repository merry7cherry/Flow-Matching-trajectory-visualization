"""Training routines for Flow Matching models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..flows.standard import FlowMatchingBatch, StandardFlowMatching
from ..training.objectives import FlowMatchingObjective


@dataclass
class TrainingConfig:
    batch_size: int = 1024
    steps: int = 5000
    log_every: int = 200


class FlowMatchingTrainer:
    """Encapsulates the optimisation loop for Flow Matching."""

    def __init__(
        self,
        flow: StandardFlowMatching,
        dataset_sampler: Callable[[int, Optional[torch.device], Optional[torch.dtype]], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        objective: Optional[nn.Module] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.flow = flow
        self.dataset_sampler = dataset_sampler
        self.optimizer = optimizer
        self.objective = objective or FlowMatchingObjective()
        self.config = config or TrainingConfig()
        self.device = device or flow.device
        self.dtype = dtype or flow.dtype
        self.flow.to(self.device, self.dtype)
        self.objective.to(self.device)

    def _sample_batch(self) -> FlowMatchingBatch:
        x1 = self.dataset_sampler(self.config.batch_size, self.device, self.dtype)
        return self.flow.make_batch(x1)

    def train(self) -> list[float]:
        self.flow.train()
        losses: list[float] = []
        for step in range(1, self.config.steps + 1):
            batch = self._sample_batch()
            prediction = self.flow.velocity(batch.xt, batch.t)
            loss = self.objective(prediction, batch.target_velocity)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if step % self.config.log_every == 0 or step == 1:
                losses.append(loss.item())
        self.flow.eval()
        return losses
