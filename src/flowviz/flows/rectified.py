from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from ..config import IntegratorConfig, RectifiedFlowConfig
from ..data.base import EmpiricalPairDataset, PairDataset
from ..simulation.integrators import EulerIntegrator
from .linear import LinearInterpolationFlow


@dataclass
class RectifiedFlowDataset:
    dataset: EmpiricalPairDataset
    x0: torch.Tensor
    x1: torch.Tensor


class RectifiedFlowBuilder:
    """Generate rectified flow datasets using a trained base flow model."""

    def __init__(
        self,
        base_objective: LinearInterpolationFlow,
        integrator_config: IntegratorConfig,
        device: torch.device,
    ) -> None:
        self.base_objective = base_objective
        self.integrator_config = integrator_config
        self.device = device

    @torch.no_grad()
    def build_dataset(
        self,
        model: torch.nn.Module,
        dataset: PairDataset,
        config: RectifiedFlowConfig,
    ) -> RectifiedFlowDataset:
        model.eval()
        integrator = EulerIntegrator(self.integrator_config.num_steps)

        samples_x0 = []
        samples_x1 = []

        remaining = config.num_samples
        while remaining > 0:
            batch_size = min(config.batch_size, remaining)
            x0 = dataset.sample_base(batch_size, self.device)
            trajectory, _ = integrator.integrate(model, x0, self.device)
            x1 = trajectory[-1]
            samples_x0.append(x0.cpu())
            samples_x1.append(x1.cpu())
            remaining -= batch_size

        x0_all = torch.cat(samples_x0, dim=0)
        x1_all = torch.cat(samples_x1, dim=0)
        empirical = EmpiricalPairDataset((x0_all, x1_all), seed=dataset.seed)
        return RectifiedFlowDataset(dataset=empirical, x0=x0_all, x1=x1_all)

    @torch.no_grad()
    def build_ground_truth(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        integrator = EulerIntegrator(self.integrator_config.num_steps)
        trajectory, times = integrator.integrate(model, x0, self.device)
        return trajectory[-1], times


__all__ = ["RectifiedFlowBuilder", "RectifiedFlowDataset"]
