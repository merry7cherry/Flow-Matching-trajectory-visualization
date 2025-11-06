from __future__ import annotations

import torch

from ..config import VariationalModifiedMeanFlowConfig


def logit_normal_timestep_sample(
    mean: float,
    std: float,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample timesteps in (0, 1) via a logit-normal distribution."""

    normal_samples = torch.randn(num_samples, device=device) * std + mean
    timesteps = torch.sigmoid(normal_samples)
    return timesteps.view(num_samples, 1)


def sample_two_timesteps_t_r_v1(
    config: VariationalModifiedMeanFlowConfig,
    num_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) pairs with the constraint that t >= r."""

    t = logit_normal_timestep_sample(config.p_mean_t, config.p_std_t, num_samples, device)
    r = logit_normal_timestep_sample(config.p_mean_r, config.p_std_r, num_samples, device)

    prob = torch.rand(num_samples, 1, device=device)
    mask = prob < 1 - config.ratio
    r = torch.where(mask, t, r)

    r = torch.minimum(t, r)
    return t, r


__all__ = ["sample_two_timesteps_t_r_v1"]
