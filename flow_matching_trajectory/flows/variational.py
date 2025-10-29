"""Variational Flow Matching implementation."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from ..data.synthetic import SyntheticBatch
from ..models.mlp import TimeConditionedMLP
from ..models.vae import ConditionalLatentEncoder
from .base import FlowMatchingVariant, FlowTrainingStep


class VariationalFlowMatching(FlowMatchingVariant):
    """Variational Flow Matching with a latent code sampled from a VAE."""

    def __init__(
        self,
        dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        kl_weight: float = 1e-3,
    ) -> None:
        velocity_model = TimeConditionedMLP(
            in_dim=dim + latent_dim,
            hidden_dims=[hidden_dim] * hidden_layers,
            out_dim=dim,
        )
        super().__init__(velocity_model, name="variational_flow_matching")
        self.encoder = ConditionalLatentEncoder(
            input_dim=dim * 3,
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim] * hidden_layers,
        )
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    @property
    def parameters(self):  # type: ignore[override]
        return list(self.velocity_model.parameters()) + list(self.encoder.parameters())

    def _additional_modules(self) -> Iterable[torch.nn.Module]:  # type: ignore[override]
        return [self.encoder]

    def compute_training_step(self, batch: SyntheticBatch) -> FlowTrainingStep:
        encoder_input = torch.cat([batch.x0, batch.x1, batch.xt], dim=-1)
        mean, logvar = self.encoder(encoder_input, batch.t)
        z = ConditionalLatentEncoder.reparameterize(mean, logvar)
        pred = self.predict_velocity(batch.xt, batch.t, z)
        target = batch.target_velocity
        mse = F.mse_loss(pred, target)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = mse + self.kl_weight * kl
        metrics = {"loss": loss.detach(), "mse": mse.detach(), "kl": kl.detach()}
        return FlowTrainingStep(loss=loss, metrics=metrics)

    def predict_velocity(
        self, xt: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("Variational Flow Matching requires a latent context.")
        return self.velocity_model(torch.cat([xt, context], dim=-1), t)

    def sample_inference_context(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)
