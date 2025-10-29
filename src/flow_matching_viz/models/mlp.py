"""Neural network architectures used in the project."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class TimeConditionedMLP(nn.Module):
    """A simple multilayer perceptron conditioned on time."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] | Iterable[int] = (128, 128),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if input_dim < 2:
            raise ValueError("The input dimension must include both space and time components.")
        layers: list[nn.Module] = []
        last_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(last_dim, width))
            layers.append(activation())
            last_dim = width
        layers.append(nn.Linear(last_dim, input_dim - 1))
        self.network = nn.Sequential(*layers)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(xt)
