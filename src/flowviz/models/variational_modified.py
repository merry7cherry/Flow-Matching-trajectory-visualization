from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLP


class VariationalModifiedEncoder(nn.Module):
    """Encode (x0, x1, xt, t, h) into latent distribution parameters."""

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        hidden_sizes: tuple[int, ...] | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or (128, 128)
        # The encoder observes the source, target, interpolant and two time scalars.
        input_dim = 3 * dim + 2
        layer_sizes = [input_dim, *hidden_sizes, 2 * latent_dim]
        self.net = MLP(layer_sizes, activation=activation)
        self.latent_dim = latent_dim

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((x0, x1, xt, t, h), dim=1)
        stats = self.net(inputs)
        mean, logvar = stats.chunk(2, dim=1)
        return mean, logvar


class VariationalModifiedVelocityMLP(nn.Module):
    """Velocity network conditioned on latent code and two time inputs."""

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        hidden_sizes: tuple[int, ...] | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or (128, 128, 128)
        # Inputs consist of the state, primary time, auxiliary time and latent code.
        input_dim = dim + 2 + latent_dim
        layer_sizes = [input_dim, *hidden_sizes, dim]
        self.mlp = MLP(layer_sizes, activation=activation)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        t_flat = t.view(t.shape[0], -1)
        h_flat = h.view(h.shape[0], -1)
        z_flat = z.view(z.shape[0], -1)
        return self.mlp(torch.cat((x_flat, t_flat, h_flat, z_flat), dim=1))


__all__ = ["VariationalModifiedEncoder", "VariationalModifiedVelocityMLP"]
