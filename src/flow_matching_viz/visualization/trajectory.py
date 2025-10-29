"""Trajectory sampling helpers."""

from __future__ import annotations

from typing import Tuple

import torch

from ..flows.standard import StandardFlowMatching


@torch.no_grad()
def sample_trajectories(
    flow: StandardFlowMatching,
    batch_size: int,
    num_steps: int = 100,
    method: str = "heun",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample trajectories by rolling out the learned velocity field."""

    initial_state = flow.sample_base(batch_size)
    trajectories = flow.rollout(initial_state, num_steps=num_steps, method=method)
    times = torch.linspace(0.0, 1.0, num_steps + 1, device=flow.device, dtype=flow.dtype)
    return times.cpu(), trajectories.cpu()
