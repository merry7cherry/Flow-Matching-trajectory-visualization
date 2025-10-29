"""Neural network architectures used for velocity and encoder models."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Embed scalar time inputs using sinusoidal features."""

    def __init__(self, embedding_dim: int = 32) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("`embedding_dim` must be even for sinusoidal embeddings.")
        self.embedding_dim = embedding_dim
        frequencies = torch.exp(torch.linspace(0.0, 4.0, embedding_dim // 2))
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        t = t.unsqueeze(-1)
        angles = t * self.frequencies * torch.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TimeConditionedMLP(nn.Module):
    """Simple MLP that concatenates position and time embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        time_embedding_dim: int = 32,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        layer_dims: List[int] = [in_dim + time_embedding_dim] + list(hidden_dims) + [out_dim]

        layers: List[nn.Module] = []
        for i in range(len(layer_dims) - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        t_emb = self.time_embed(t)
        return self.network(torch.cat([x, t_emb], dim=-1))
