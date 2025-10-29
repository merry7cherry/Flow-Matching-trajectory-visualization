"""Rectified Flow variant building on a pre-trained Flow Matching model."""

from __future__ import annotations

from typing import Dict

import torch

from ..data.synthetic import SyntheticPairDataset
from ..utils.integration import euler_integrate
from .linear import LinearFlowMatching


class RectifiedFlowMatching(LinearFlowMatching):
    """Rectified Flow variant using samples generated via a base model."""

    def __init__(self, dim: int, hidden_dim: int = 128, hidden_layers: int = 3) -> None:
        super().__init__(dim=dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.name = "rectified_flow"


@torch.no_grad()
def generate_rectified_pairs(
    base_model: LinearFlowMatching,
    dataset: SyntheticPairDataset,
    num_samples: int,
    integration_steps: int,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """Generate rectified flow training pairs using a pre-trained model."""

    base_model.eval()
    collected_x0 = []
    collected_x1 = []
    times = torch.linspace(0.0, 1.0, integration_steps + 1, device=device)

    total = 0
    while total < num_samples:
        current_batch = min(batch_size, num_samples - total)
        batch = dataset.sample(current_batch)
        x0 = batch.x0.to(device)

        def velocity_fn(state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return base_model.predict_velocity(state, t)

        trajectory, _ = euler_integrate(velocity_fn, x0, times)
        z1 = trajectory[-1]
        collected_x0.append(x0.cpu())
        collected_x1.append(z1.cpu())
        total += current_batch

    return {"x0": torch.cat(collected_x0, dim=0)[:num_samples], "x1": torch.cat(collected_x1, dim=0)[:num_samples]}
