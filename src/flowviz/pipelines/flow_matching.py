from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.func import jvp

from ..config import (
    IntegratorConfig,
    RectifiedFlowConfig,
    TrainingConfig,
    VariationalFlowConfig,
    VariationalMeanFlowConfig,
)
from ..data.base import PairDataset
from ..flows.linear import LinearInterpolationFlow
from ..flows.rectified import RectifiedFlowBuilder
from ..flows.variational import VariationalFlowObjective
from ..models.mlp import VelocityMLP
from ..models.variational import VariationalEncoder, VariationalVelocityMLP
from ..simulation.integrators import EulerIntegrator
from ..training.trainer import TrainingHistory, train_model


@dataclass
class ExperimentArtifacts:
    model: torch.nn.Module
    history: TrainingHistory


@dataclass
class VariationalTrainingHistory:
    total_losses: List[float]
    matching_losses: List[float]
    kl_losses: List[float]


@dataclass
class VariationalExperimentArtifacts:
    velocity_model: VariationalVelocityMLP
    encoder: VariationalEncoder
    history: VariationalTrainingHistory


@dataclass
class VariationalMeanExperimentArtifacts:
    velocity_model: VariationalVelocityMLP
    encoder: VariationalEncoder
    history: VariationalTrainingHistory


def train_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    hidden_sizes: list[int] | None = None,
) -> ExperimentArtifacts:
    model = VelocityMLP(dim=dataset.dim, hidden_sizes=hidden_sizes)
    objective = LinearInterpolationFlow()
    history = train_model(model, dataset, objective, training_config)
    return ExperimentArtifacts(model=model, history=history)


def generate_ground_truth(
    x0: torch.Tensor,
    x1: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    times = torch.linspace(0.0, 1.0, num_steps + 1, device=x0.device)
    ground_truth = []
    for t in times:
        gt = (1.0 - t) * x0 + t * x1
        ground_truth.append(gt)
    return torch.stack(ground_truth, dim=0), times


def compute_model_trajectories(
    model: torch.nn.Module,
    x0: torch.Tensor,
    device: torch.device,
    integrator_config: IntegratorConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    integrator = EulerIntegrator(num_steps=integrator_config.num_steps)
    with torch.no_grad():
        trajectory, times = integrator.integrate(model, x0.to(device), device)
    return trajectory, times


def train_rectified_flow(
    base_model: torch.nn.Module,
    base_dataset: PairDataset,
    training_config: TrainingConfig,
    integrator_config: IntegratorConfig,
    rectified_config: RectifiedFlowConfig,
) -> Tuple[ExperimentArtifacts, RectifiedFlowBuilder]:
    objective = LinearInterpolationFlow()
    builder = RectifiedFlowBuilder(objective, integrator_config, torch.device(training_config.device))
    rectified_dataset = builder.build_dataset(base_model, base_dataset, rectified_config).dataset
    rectified_model = VelocityMLP(dim=base_dataset.dim)
    history = train_model(rectified_model, rectified_dataset, objective, training_config)
    return ExperimentArtifacts(model=rectified_model, history=history), builder


def train_variational_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    variational_config: VariationalFlowConfig,
) -> VariationalExperimentArtifacts:
    device = torch.device(training_config.device)
    objective = VariationalFlowObjective()

    velocity_model = VariationalVelocityMLP(
        dim=dataset.dim,
        latent_dim=variational_config.latent_dim,
        hidden_sizes=variational_config.velocity_hidden_sizes,
    ).to(device)

    encoder = VariationalEncoder(
        dim=dataset.dim,
        latent_dim=variational_config.latent_dim,
        hidden_sizes=variational_config.encoder_hidden_sizes,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(velocity_model.parameters()) + list(encoder.parameters()),
        lr=training_config.learning_rate,
    )

    total_history: List[float] = []
    matching_history: List[float] = []
    kl_history: List[float] = []

    for _ in range(training_config.epochs):
        epoch_total = 0.0
        epoch_matching = 0.0
        epoch_kl = 0.0

        for _ in range(training_config.steps_per_epoch):
            batch = dataset.sample_pairs(training_config.batch_size, device)
            t = objective.sample_time(training_config.batch_size, device)
            xt = objective.interpolate(batch.x0, batch.x1, t)
            target_velocity = objective.target_velocity(batch.x0, batch.x1, xt, t)

            mean, logvar = encoder(batch.x0, batch.x1, xt, t)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + std * eps

            predicted_velocity = velocity_model(xt, t, z)

            matching_loss = F.mse_loss(predicted_velocity, target_velocity)

            kl_loss = 0.5 * torch.mean(
                torch.sum(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar, dim=1)
            )

            loss = (
                variational_config.matching_weight * matching_loss
                + variational_config.kl_weight * kl_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_matching += matching_loss.item()
            epoch_kl += kl_loss.item()

        steps = max(1, training_config.steps_per_epoch)
        total_history.append(epoch_total / steps)
        matching_history.append(epoch_matching / steps)
        kl_history.append(epoch_kl / steps)

    history = VariationalTrainingHistory(
        total_losses=total_history,
        matching_losses=matching_history,
        kl_losses=kl_history,
    )

    return VariationalExperimentArtifacts(
        velocity_model=velocity_model,
        encoder=encoder,
        history=history,
    )


def train_variational_mean_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    variational_config: VariationalMeanFlowConfig,
) -> VariationalMeanExperimentArtifacts:
    """Train the variational mean flow objective using forward-mode autodiff."""
    device = torch.device(training_config.device)
    objective = VariationalFlowObjective()

    velocity_model = VariationalVelocityMLP(
        dim=dataset.dim,
        latent_dim=variational_config.latent_dim,
        hidden_sizes=variational_config.velocity_hidden_sizes,
    ).to(device)

    encoder = VariationalEncoder(
        dim=dataset.dim,
        latent_dim=variational_config.latent_dim,
        hidden_sizes=variational_config.encoder_hidden_sizes,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(velocity_model.parameters()) + list(encoder.parameters()),
        lr=training_config.learning_rate,
    )

    total_history: List[float] = []
    matching_history: List[float] = []
    kl_history: List[float] = []

    velocity_model.train()
    encoder.train()

    for _ in range(training_config.epochs):
        epoch_total = 0.0
        epoch_matching = 0.0
        epoch_kl = 0.0

        for _ in range(training_config.steps_per_epoch):
            batch = dataset.sample_pairs(training_config.batch_size, device)
            time_samples = objective.sample_time(training_config.batch_size, device)
            interpolated_state = objective.interpolate(batch.x0, batch.x1, time_samples)
            linear_velocity = batch.x1 - batch.x0

            latent_noise = torch.randn(
                interpolated_state.shape[0],
                variational_config.latent_dim,
                device=device,
                dtype=interpolated_state.dtype,
            )

            unit_time_tangent = torch.ones_like(time_samples)

            zero_x0_tangent = torch.zeros_like(batch.x0)
            zero_x1_tangent = torch.zeros_like(batch.x1)

            def sample_latent_with_encoder(
                x0_in: torch.Tensor,
                x1_in: torch.Tensor,
                xt_in: torch.Tensor,
                t_in: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                mean, logvar = encoder(x0_in, x1_in, xt_in, t_in)
                std = torch.exp(0.5 * logvar)
                latent_sample = mean + std * latent_noise
                return latent_sample, mean, logvar

            (latent_sample, latent_mean, latent_logvar), (latent_time_derivative, _, _) = jvp(
                sample_latent_with_encoder,
                (batch.x0, batch.x1, interpolated_state, time_samples),
                (zero_x0_tangent, zero_x1_tangent, linear_velocity, unit_time_tangent),
            )

            def latent_conditioned_velocity(
                xt_in: torch.Tensor,
                t_in: torch.Tensor,
                z_in: torch.Tensor,
            ) -> torch.Tensor:
                return velocity_model(xt_in, t_in, z_in)

            predicted_velocity, velocity_time_derivative = jvp(
                latent_conditioned_velocity,
                (interpolated_state, time_samples, latent_sample),
                (linear_velocity, unit_time_tangent, latent_time_derivative),
            )

            flow_matching_target = (
                linear_velocity + (1.0 - time_samples) * velocity_time_derivative
            ).detach()

            matching_loss = F.mse_loss(predicted_velocity, flow_matching_target)

            kl_loss = 0.5 * torch.mean(
                torch.sum(torch.exp(latent_logvar) + latent_mean.pow(2) - 1.0 - latent_logvar, dim=1)
            )

            loss = (
                variational_config.matching_weight * matching_loss
                + variational_config.kl_weight * kl_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_matching += matching_loss.item()
            epoch_kl += kl_loss.item()

        steps = max(1, training_config.steps_per_epoch)
        total_history.append(epoch_total / steps)
        matching_history.append(epoch_matching / steps)
        kl_history.append(epoch_kl / steps)

    history = VariationalTrainingHistory(
        total_losses=total_history,
        matching_losses=matching_history,
        kl_losses=kl_history,
    )

    return VariationalMeanExperimentArtifacts(
        velocity_model=velocity_model,
        encoder=encoder,
        history=history,
    )


class _VariationalTrajectoryWrapper(torch.nn.Module):
    """Wrap the variational velocity network for trajectory integration."""

    def __init__(self, model: VariationalVelocityMLP, z: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.z = z

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x, t, self.z)


class _VariationalMeanTrajectoryWrapper(torch.nn.Module):
    """Wrapper that reuses the variational velocity model for VMF sampling."""

    def __init__(self, model: VariationalVelocityMLP, z: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.z = z

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x, t, self.z)


def compute_variational_trajectories(
    model: VariationalVelocityMLP,
    x0: torch.Tensor,
    device: torch.device,
    integrator_config: IntegratorConfig,
    variational_config: VariationalFlowConfig,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    integrator = EulerIntegrator(num_steps=integrator_config.num_steps)
    batch_size = x0.shape[0]
    if generator is not None:
        z = torch.randn(
            batch_size,
            variational_config.latent_dim,
            device=device,
            generator=generator,
        )
    else:
        z = torch.randn(batch_size, variational_config.latent_dim, device=device)
    wrapper = _VariationalTrajectoryWrapper(model, z)
    with torch.no_grad():
        trajectory, times = integrator.integrate(wrapper, x0.to(device), device)
    return trajectory, times


def compute_variational_mean_trajectories(
    model: VariationalVelocityMLP,
    encoder: VariationalEncoder,
    x0: torch.Tensor,
    device: torch.device,
    integrator_config: IntegratorConfig,
    variational_config: VariationalMeanFlowConfig,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate VMF trajectories by sampling the latent code from the prior."""
    model.eval()
    encoder.eval()
    integrator = EulerIntegrator(num_steps=integrator_config.num_steps)
    x0_device = x0.to(device)
    batch_size = x0.shape[0]
    if generator is not None:
        z = torch.randn(
            batch_size,
            variational_config.latent_dim,
            device=device,
            dtype=x0_device.dtype,
            generator=generator,
        )
    else:
        z = torch.randn(
            batch_size,
            variational_config.latent_dim,
            device=device,
            dtype=x0_device.dtype,
        )
    wrapper = _VariationalMeanTrajectoryWrapper(model, z)
    with torch.no_grad():
        trajectory, times = integrator.integrate(wrapper, x0_device, device)
    return trajectory, times


__all__ = [
    "ExperimentArtifacts",
    "VariationalExperimentArtifacts",
    "VariationalMeanExperimentArtifacts",
    "VariationalTrainingHistory",
    "train_flow_matching",
    "train_rectified_flow",
    "train_variational_flow_matching",
    "train_variational_mean_flow_matching",
    "generate_ground_truth",
    "compute_model_trajectories",
    "compute_variational_trajectories",
    "compute_variational_mean_trajectories",
]
