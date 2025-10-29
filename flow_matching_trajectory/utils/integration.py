"""Numerical integration utilities for simulating flow trajectories."""

from __future__ import annotations

from typing import Callable, Tuple

import torch


def euler_integrate(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    times: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate an ODE forward in time using the explicit Euler method.

    Args:
        velocity_fn: Callable returning the velocity for the current state and
            time. The callable signature is ``velocity_fn(state, t)`` where
            ``state`` has shape ``(batch, dim)`` and ``t`` has shape
            ``(batch, )``.
        initial_state: Tensor containing the initial state ``Z_0``.
        times: Monotonically increasing 1D tensor of integration times.

    Returns:
        A tuple ``(trajectory, velocities)`` containing the simulated states and
        the corresponding velocities for each integration step. ``trajectory``
        has shape ``(len(times), batch, dim)`` where the first entry equals the
        initial state. ``velocities`` has shape ``(len(times) - 1, batch, dim)``.
    """

    if times.ndim != 1:
        raise ValueError("`times` must be a 1D tensor of integration steps.")

    num_steps = times.numel()
    batch_size, dim = initial_state.shape
    device = initial_state.device

    trajectory = torch.empty((num_steps, batch_size, dim), device=device)
    velocities = torch.empty((num_steps - 1, batch_size, dim), device=device)
    trajectory[0] = initial_state

    for i in range(num_steps - 1):
        t0, t1 = times[i], times[i + 1]
        dt = t1 - t0
        t_batch = torch.full((batch_size,), t0, device=device)
        vel = velocity_fn(trajectory[i], t_batch)
        velocities[i] = vel
        trajectory[i + 1] = trajectory[i] + dt * vel

    return trajectory, velocities
