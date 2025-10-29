"""Variational encoder module used by the VFM variant."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn

from .mlp import SinusoidalTimeEmbedding


class ConditionalLatentEncoder(nn.Module):
    """Encoder predicting the parameters of q(z | x0, x1, xt, t)."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Iterable[int],
        time_embedding_dim: int = 32,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        dims = [input_dim + time_embedding_dim] + list(hidden_dims)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.to_mean = nn.Linear(dims[-1], latent_dim)
        self.to_logvar = nn.Linear(dims[-1], latent_dim)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(torch.cat([inputs, self.time_embed(t)], dim=-1))
        mean = self.to_mean(features)
        logvar = self.to_logvar(features)
        return mean, logvar

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
