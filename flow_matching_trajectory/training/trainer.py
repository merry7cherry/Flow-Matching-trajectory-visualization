"""Training utilities shared across Flow Matching variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import optim

from ..data.synthetic import SyntheticPairDataset
from ..flows.base import FlowMatchingVariant
from ..utils.logging import configure_logging
from .types import TrainingHistory


@dataclass
class TrainerConfig:
    batch_size: int = 256
    num_steps: int = 5_000
    learning_rate: float = 1e-3
    log_interval: int = 100
    device: torch.device = torch.device("cpu")


class FlowMatchingTrainer:
    """Simple trainer performing gradient descent on Flow Matching variants."""

    def __init__(
        self,
        variant: FlowMatchingVariant,
        dataset: SyntheticPairDataset,
        config: TrainerConfig,
        logger_name: Optional[str] = None,
    ) -> None:
        self.variant = variant.to(config.device)
        self.dataset = dataset
        self.config = config
        self.logger = configure_logging(name=logger_name or variant.name)
        self.optimizer = optim.Adam(self.variant.parameters, lr=config.learning_rate)
        self.history: TrainingHistory = TrainingHistory()

    def train(self) -> TrainingHistory:
        self.variant.train()
        for step in range(1, self.config.num_steps + 1):
            batch = self.dataset.sample(self.config.batch_size)
            batch = batch  # already on device via dataset
            self.optimizer.zero_grad()
            output = self.variant.compute_training_step(batch)
            output.loss.backward()
            self.optimizer.step()

            self.history.update(step, output.metrics)
            if step % self.config.log_interval == 0 or step == 1:
                metric_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in output.metrics.items())
                self.logger.info("Step %d: %s", step, metric_str)
        return self.history


