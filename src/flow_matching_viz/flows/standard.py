"""Implementation of the standard Flow Matching setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..utils.samplers import StandardNormalSampler, UniformTimeSampler


@dataclass
class FlowMatchingBatch:
    x0: torch.Tensor
    x1: torch.Tensor
    t: torch.Tensor
    xt: torch.Tensor
    target_velocity: torch.Tensor


class StandardFlowMatching:
    """Standard Flow Matching with straight interpolation between endpoints."""

    def __init__(
        self,
        velocity_model: nn.Module,
        data_dim: int,
        base_sampler: Optional[StandardNormalSampler] = None,
        time_sampler: Optional[UniformTimeSampler] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.velocity_model = velocity_model
        self.data_dim = data_dim
        self.base_sampler = base_sampler or StandardNormalSampler(dimension=data_dim)
        self.time_sampler = time_sampler or UniformTimeSampler()
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "StandardFlowMatching":
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.velocity_model.to(device=device, dtype=self.dtype)
        return self

    def sample_base(self, batch_size: int) -> torch.Tensor:
        return self.base_sampler.sample(batch_size, device=self.device, dtype=self.dtype)

    def sample_time(self, batch_size: int) -> torch.Tensor:
        return self.time_sampler.sample(batch_size, device=self.device, dtype=self.dtype)

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def velocity(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.shape[-1] != 1:
            raise ValueError("Time tensor must have a singleton feature dimension.")
        if xt.shape[-1] != self.data_dim:
            raise ValueError("Spatial tensor does not match the configured dimensionality.")
        t_expanded = t.expand(-1, self.data_dim)
        model_input = torch.cat([xt, t_expanded], dim=-1)
        return self.velocity_model(model_input)

    def make_batch(self, x1: torch.Tensor) -> FlowMatchingBatch:
        batch_size = x1.shape[0]
        x0 = self.sample_base(batch_size)
        t = self.sample_time(batch_size)
        xt = self.interpolate(x0, x1, t)
        target_vel = self.target_velocity(x0, x1)
        return FlowMatchingBatch(x0=x0, x1=x1, t=t, xt=xt, target_velocity=target_vel)

    def eval(self) -> None:
        self.velocity_model.eval()

    def train(self) -> None:
        self.velocity_model.train()

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        num_steps: int = 100,
        method: str = "heun",
    ) -> torch.Tensor:
        x = self.sample_base(num_samples)
        trajectories = self.rollout(x, num_steps=num_steps, method=method)
        return trajectories[:, -1]

    @torch.no_grad()
    def rollout(
        self,
        x0: torch.Tensor,
        num_steps: int = 100,
        method: str = "heun",
    ) -> torch.Tensor:
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=self.dtype)
        x = x0.to(device=self.device, dtype=self.dtype)
        states = [x]
        for i in range(num_steps):
            t = times[i]
            dt = times[i + 1] - times[i]
            if method == "euler":
                velocity = self.velocity(x, torch.full((x.shape[0], 1), t, device=self.device, dtype=self.dtype))
                x = x + velocity * dt
            elif method == "heun":
                t_tensor = torch.full((x.shape[0], 1), t, device=self.device, dtype=self.dtype)
                v0 = self.velocity(x, t_tensor)
                x_proposed = x + v0 * dt
                v1 = self.velocity(x_proposed, torch.full_like(t_tensor, t + dt))
                x = x + 0.5 * (v0 + v1) * dt
            else:
                raise ValueError(f"Unknown integration method: {method}.")
            states.append(x)
        return torch.stack(states, dim=1)
