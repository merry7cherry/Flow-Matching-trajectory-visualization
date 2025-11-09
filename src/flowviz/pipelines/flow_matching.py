from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch.func import jvp

from ..config import (
    IntegratorConfig,
    MeanFlowConfig,
    RectifiedFlowConfig,
    TrainingConfig,
    VariationalFlowConfig,
    VariationalForwardMeanFlowConfig,
    VariationalForwardMeanFlowModifiedConfig,
    VariationalMeanFlowConfig,
    VariationalMeanFlowModifiedConfig,
)
from ..data.base import PairDataset
from ..flows.linear import LinearInterpolationFlow
from ..flows.rectified import RectifiedFlowBuilder
from ..flows.variational import VariationalFlowObjective
from ..flows.time_sampling import sample_two_timesteps_t_r_v1
from ..models.mlp import MeanVelocityMLP, VelocityMLP
from ..models.variational import (
    VariationalEncoder,
    VariationalForwardEncoder,
    VariationalMeanEncoder,
    VariationalMeanVelocityMLP,
    VariationalVelocityMLP,
)
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
class VariationalMeanFlowExperimentArtifacts:
    velocity_model: VariationalMeanVelocityMLP
    encoder: VariationalMeanEncoder
    history: VariationalTrainingHistory


@dataclass
class VariationalMeanFlowModifiedExperimentArtifacts:
    velocity_model: VariationalMeanVelocityMLP
    encoder: VariationalForwardEncoder
    history: VariationalTrainingHistory


@dataclass
class VariationalForwardMeanExperimentArtifacts:
    velocity_model: VariationalVelocityMLP
    encoder: VariationalEncoder
    history: VariationalTrainingHistory


@dataclass
class VariationalForwardMeanModifiedExperimentArtifacts:
    velocity_model: VariationalVelocityMLP
    encoder: VariationalForwardEncoder
    history: VariationalTrainingHistory


@dataclass
class MeanFlowExperimentArtifacts:
    model: MeanVelocityMLP
    history: TrainingHistory


def train_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    hidden_sizes: list[int] | None = None,
) -> ExperimentArtifacts:
    model = VelocityMLP(dim=dataset.dim, hidden_sizes=hidden_sizes)
    objective = LinearInterpolationFlow()
    history = train_model(model, dataset, objective, training_config)
    return ExperimentArtifacts(model=model, history=history)


def train_mean_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    mean_config: MeanFlowConfig,
) -> MeanFlowExperimentArtifacts:
    """Train the mean flow objective with the (t, r) time parametrization."""

    device = torch.device(training_config.device)
    model = MeanVelocityMLP(
        dim=dataset.dim, hidden_sizes=list(mean_config.velocity_hidden_sizes)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    losses: List[float] = []

    model.train()
    for _ in range(training_config.epochs):
        epoch_loss = 0.0
        for _ in range(training_config.steps_per_epoch):
            batch = dataset.sample_pairs(training_config.batch_size, device)
            t_raw, r_raw = sample_two_timesteps_t_r_v1(
                mean_config, training_config.batch_size, device
            )
            t = t_raw.view(-1, 1)
            r = r_raw.view(-1, 1)

            x0 = batch.x0
            x1 = batch.x1
            z = (1.0 - t) * x1 + t * x0
            v = x0 - x1

            def u_func(z_in: torch.Tensor, t_in: torch.Tensor, r_in: torch.Tensor) -> torch.Tensor:
                h_in = t_in - r_in
                return model(z_in, t_in, h_in)

            dtdt = torch.ones_like(t)
            drdt = torch.zeros_like(r)

            predicted_velocity, velocity_time_derivative = jvp(
                u_func,
                (z, t, r),
                (v, dtdt, drdt),
            )

            u_target = (v - (t - r) * velocity_time_derivative).detach()
            loss_terms = (predicted_velocity - u_target) ** 2
            loss_terms = loss_terms.view(loss_terms.shape[0], -1).sum(dim=1)
            adaptive_weight = (loss_terms.detach() + mean_config.norm_eps) ** mean_config.norm_p
            loss = torch.mean(loss_terms / adaptive_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        steps = max(1, training_config.steps_per_epoch)
        losses.append(epoch_loss / steps)

    history = TrainingHistory(losses=losses)
    return MeanFlowExperimentArtifacts(model=model, history=history)


def train_variational_mean_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    variational_config: VariationalMeanFlowConfig,
) -> VariationalMeanFlowExperimentArtifacts:
    """Train the variational mean flow objective with latent conditioning on (x0, x1, xt, t, h)."""

    device = torch.device(training_config.device)
    velocity_model = VariationalMeanVelocityMLP(
        dim=dataset.dim,
        latent_dim=variational_config.latent_dim,
        hidden_sizes=variational_config.velocity_hidden_sizes,
    ).to(device)

    encoder = VariationalMeanEncoder(
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
            t_raw, r_raw = sample_two_timesteps_t_r_v1(
                variational_config, training_config.batch_size, device
            )
            t = t_raw.view(-1, 1)
            r = r_raw.view(-1, 1)

            h = t - r
            x0 = batch.x0
            x1 = batch.x1
            interpolated_state = (1.0 - t) * x1 + t * x0
            linear_velocity = x0 - x1

            latent_noise = torch.randn(
                interpolated_state.shape[0],
                variational_config.latent_dim,
                device=device,
                dtype=interpolated_state.dtype,
            )

            zero_x0_tangent = torch.zeros_like(x0)
            zero_x1_tangent = torch.zeros_like(x1)
            dtdt = torch.ones_like(t)
            drdt = torch.zeros_like(r)

            def sample_latent_with_encoder(
                x0_in: torch.Tensor,
                x1_in: torch.Tensor,
                xt_in: torch.Tensor,
                t_in: torch.Tensor,
                r_in: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                h_in = t_in - r_in
                mean, logvar = encoder(x0_in, x1_in, xt_in, t_in, h_in)
                std = torch.exp(0.5 * logvar)
                latent_sample = mean + std * latent_noise
                return latent_sample, mean, logvar

            (latent_sample, latent_mean, latent_logvar), (
                latent_time_derivative,
                _,
                _,
            ) = jvp(
                sample_latent_with_encoder,
                (x0, x1, interpolated_state, t, r),
                (
                    zero_x0_tangent,
                    zero_x1_tangent,
                    linear_velocity,
                    dtdt,
                    drdt,
                ),
            )

            def latent_conditioned_velocity(
                z_in: torch.Tensor,
                t_in: torch.Tensor,
                r_in: torch.Tensor,
                latent_in: torch.Tensor,
            ) -> torch.Tensor:
                h_in = t_in - r_in
                return velocity_model(z_in, t_in, h_in, latent_in)

            predicted_velocity, velocity_time_derivative = jvp(
                latent_conditioned_velocity,
                (interpolated_state, t, r, latent_sample),
                (linear_velocity, dtdt, drdt, latent_time_derivative),
            )

            flow_target = (linear_velocity - h * velocity_time_derivative).detach()
            squared_error = (predicted_velocity - flow_target) ** 2
            squared_error = squared_error.view(squared_error.shape[0], -1).sum(dim=1)
            adaptive_weight = (squared_error.detach() + variational_config.norm_eps) ** variational_config.norm_p
            matching_loss = torch.mean(squared_error / adaptive_weight)

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

    return VariationalMeanFlowExperimentArtifacts(
        velocity_model=velocity_model,
        encoder=encoder,
        history=history,
    )


def train_variational_mean_flow_modified_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    variational_config: VariationalMeanFlowModifiedConfig,
) -> VariationalMeanFlowModifiedExperimentArtifacts:
    """Train the variational mean flow modified objective with latent conditioning."""

    device = torch.device(training_config.device)
    velocity_model = VariationalMeanVelocityMLP(
        dim=dataset.dim,
        latent_dim=variational_config.latent_dim,
        hidden_sizes=variational_config.velocity_hidden_sizes,
    ).to(device)

    encoder = VariationalForwardEncoder(
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
            t_raw, r_raw = sample_two_timesteps_t_r_v1(
                variational_config, training_config.batch_size, device
            )
            t = t_raw.view(-1, 1)
            r = r_raw.view(-1, 1)

            h = t - r
            x0 = batch.x0
            x1 = batch.x1
            interpolated_state = (1.0 - t) * x1 + t * x0
            linear_velocity = x0 - x1

            mean, logvar = encoder(x0, x1)
            std = torch.exp(0.5 * logvar)
            latent_noise = torch.randn_like(std)
            latent_sample = mean + std * latent_noise
            latent_time_derivative = torch.zeros_like(latent_sample)

            def latent_conditioned_velocity(
                z_in: torch.Tensor,
                t_in: torch.Tensor,
                r_in: torch.Tensor,
                latent_in: torch.Tensor,
            ) -> torch.Tensor:
                h_in = t_in - r_in
                return velocity_model(z_in, t_in, h_in, latent_in)

            dtdt = torch.ones_like(t)
            drdt = torch.zeros_like(r)

            predicted_velocity, velocity_time_derivative = jvp(
                latent_conditioned_velocity,
                (interpolated_state, t, r, latent_sample),
                (linear_velocity, dtdt, drdt, latent_time_derivative),
            )

            flow_target = (linear_velocity - h * velocity_time_derivative).detach()
            squared_error = (predicted_velocity - flow_target) ** 2
            squared_error = squared_error.view(squared_error.shape[0], -1).sum(dim=1)
            adaptive_weight = (squared_error.detach() + variational_config.norm_eps) ** variational_config.norm_p
            matching_loss = torch.mean(squared_error / adaptive_weight)

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

    return VariationalMeanFlowModifiedExperimentArtifacts(
        velocity_model=velocity_model,
        encoder=encoder,
        history=history,
    )


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


def generate_cheat_ground_truth(
    ground_truth: torch.Tensor,
    ratio: float,
    curvature: float,
    *,
    target_noise_ratio: float = 0.0,
    target_noise_strength: float = 0.0,
    trajectory_noise_ratio: float = 0.0,
    trajectory_noise_strength: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create curved variants of ground-truth trajectories for cheat visualizations.

    The transformation is applied in three phases:

    1. Optionally perturb a subset of target endpoints with random noise.
    2. Convert a fraction of trajectories into quadratic Bézier curves.
    3. Optionally add noise along the interior points of the resulting curves.
    """

    if ground_truth.ndim != 3:
        raise ValueError("ground_truth must have shape (steps, batch, dim)")
    if ground_truth.shape[-1] != 2:
        raise ValueError("cheat ground truth is only supported for 2D trajectories")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be within [0, 1]")
    if curvature < 0.0:
        raise ValueError("curvature must be non-negative")
    if not 0.0 <= target_noise_ratio <= 1.0:
        raise ValueError("target_noise_ratio must be within [0, 1]")
    if target_noise_strength < 0.0:
        raise ValueError("target_noise_strength must be non-negative")
    if not 0.0 <= trajectory_noise_ratio <= 1.0:
        raise ValueError("trajectory_noise_ratio must be within [0, 1]")
    if trajectory_noise_strength < 0.0:
        raise ValueError("trajectory_noise_strength must be non-negative")

    num_trajectories = ground_truth.shape[1]
    if num_trajectories == 0:
        return ground_truth.clone()

    cheat = ground_truth.clone()
    device = ground_truth.device
    dtype = ground_truth.dtype

    if target_noise_ratio > 0.0 and target_noise_strength > 0.0:
        num_noisy_targets = min(
            num_trajectories, math.ceil(num_trajectories * target_noise_ratio)
        )
        if num_noisy_targets > 0:
            permutation = torch.randperm(
                num_trajectories, device=device, generator=generator
            )
            target_indices = permutation[:num_noisy_targets]

            for idx_tensor in target_indices:
                idx = int(idx_tensor.item())
                start = ground_truth[0, idx]
                end = ground_truth[-1, idx]
                direction = end - start
                length = torch.norm(direction)
                if torch.isnan(length) or length <= 1e-8:
                    continue

                amplitude = target_noise_strength * length
                if amplitude <= 0:
                    continue

                noise = torch.randn(
                    (ground_truth.shape[-1],),
                    device=device,
                    generator=generator,
                    dtype=dtype,
                )
                cheat[-1, idx] = cheat[-1, idx] + noise * amplitude

    steps = ground_truth.shape[0]
    t_values = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype).view(-1, 1)
    one_minus_t = 1.0 - t_values

    if ratio > 0.0 and curvature > 0.0:
        num_curved = min(num_trajectories, math.ceil(num_trajectories * ratio))
        if num_curved > 0:
            permutation = torch.randperm(
                num_trajectories, device=device, generator=generator
            )
            selected = permutation[:num_curved]

            for idx_tensor in selected:
                idx = int(idx_tensor.item())
                start = cheat[0, idx]
                end = cheat[-1, idx]
                direction = end - start
                length = torch.norm(direction)
                if torch.isnan(length) or length <= 1e-8:
                    continue

                perp = torch.stack((-direction[1], direction[0]))
                perp_norm = perp / (torch.norm(perp) + 1e-8)

                if generator is None:
                    curvature_random = torch.rand((), device=device)
                else:
                    curvature_random = torch.rand(
                        (), device=device, generator=generator
                    )
                curvature_random = curvature_random.to(dtype=dtype)

                amplitude = curvature_random * curvature * length
                if amplitude <= 0:
                    continue

                if generator is None:
                    sign_random = torch.rand((), device=device)
                else:
                    sign_random = torch.rand((), device=device, generator=generator)
                sign = torch.where(sign_random >= 0.5, 1.0, -1.0)
                sign = sign.to(dtype=dtype)

                control = (start + end) / 2.0 + perp_norm * amplitude * sign

                curved = (
                    (one_minus_t**2) * start
                    + 2.0 * one_minus_t * t_values * control
                    + (t_values**2) * end
                )
                cheat[:, idx] = curved

    if trajectory_noise_ratio > 0.0 and trajectory_noise_strength > 0.0:
        num_noisy_trajectories = min(
            num_trajectories, math.ceil(num_trajectories * trajectory_noise_ratio)
        )
        if num_noisy_trajectories > 0 and steps > 2:
            permutation = torch.randperm(
                num_trajectories, device=device, generator=generator
            )
            trajectory_indices = permutation[:num_noisy_trajectories]

            for idx_tensor in trajectory_indices:
                idx = int(idx_tensor.item())
                start = cheat[0, idx]
                end = cheat[-1, idx]
                direction = end - start
                length = torch.norm(direction)
                if torch.isnan(length) or length <= 1e-8:
                    continue

                amplitude = trajectory_noise_strength * length
                if amplitude <= 0:
                    continue

                interior = cheat[1:-1, idx]
                noise = torch.randn(
                    interior.shape,
                    device=device,
                    generator=generator,
                    dtype=dtype,
                )
                cheat[1:-1, idx] = interior + noise * amplitude

    return cheat


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


def compute_mean_flow_trajectories(
    model: MeanVelocityMLP,
    x0: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate mean-flow trajectories using uniformly spaced inference steps.

    Args:
        model: Trained mean-flow velocity network.
        x0: Batch of base samples to transport.
        device: Torch device used for evaluation.
        steps: Number of uniform inference steps between time 0 and 1.

    Returns:
        Tuple containing the stacked trajectory and the associated time stamps.
    """

    if steps < 1:
        raise ValueError("steps must be a positive integer")

    model.eval()
    with torch.no_grad():
        base = x0.to(device)
        batch_size = base.shape[0]
        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=base.dtype)

        states = [base]
        current = base

        for idx in range(steps):
            current_time = times[idx]
            evaluation_time = 1.0 - current_time
            next_time = times[idx + 1]
            reference_time = 1.0 - next_time
            t = torch.ones((batch_size, 1), device=device, dtype=base.dtype) * evaluation_time
            r = torch.ones_like(t) * reference_time
            h = t - r

            velocity = model(current, t, h)
            dt = times[idx + 1] - current_time
            current = current - velocity * dt
            states.append(current)

        trajectory = torch.stack(states, dim=0)

    return trajectory, times


def compute_variational_mean_flow_trajectories(
    model: VariationalMeanVelocityMLP,
    x0: torch.Tensor,
    device: torch.device,
    variational_config: VariationalMeanFlowConfig,
    *,
    steps: int = 1,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate variational mean flow trajectories using latent-conditioned velocity."""

    if steps < 1:
        raise ValueError("steps must be a positive integer")

    model.eval()
    with torch.no_grad():
        base = x0.to(device)
        batch_size = base.shape[0]
        dtype = base.dtype
        if generator is not None:
            latent = torch.randn(
                batch_size,
                variational_config.latent_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            latent = torch.randn(
                batch_size,
                variational_config.latent_dim,
                device=device,
                dtype=dtype,
            )

        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

        states = [base]
        current = base

        for idx in range(steps):
            current_time = times[idx]
            evaluation_time = 1.0 - current_time
            next_time = times[idx + 1]
            reference_time = 1.0 - next_time
            t = torch.ones((batch_size, 1), device=device, dtype=dtype) * evaluation_time
            r = torch.ones_like(t) * reference_time
            h = t - r

            velocity = model(current, t, h, latent)
            dt = times[idx + 1] - current_time
            current = current - velocity * dt
            states.append(current)

        trajectory = torch.stack(states, dim=0)

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


def train_variational_forward_mean_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    variational_config: VariationalForwardMeanFlowConfig,
) -> VariationalForwardMeanExperimentArtifacts:
    """Train the variational forward mean flow objective using forward-mode autodiff."""
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

    return VariationalForwardMeanExperimentArtifacts(
        velocity_model=velocity_model,
        encoder=encoder,
        history=history,
    )


def train_variational_forward_mean_flow_modified_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    variational_config: VariationalForwardMeanFlowModifiedConfig,
) -> VariationalForwardMeanModifiedExperimentArtifacts:
    """Train the modified variational forward mean flow objective."""

    device = torch.device(training_config.device)
    objective = VariationalFlowObjective()

    velocity_model = VariationalVelocityMLP(
        dim=dataset.dim,
        latent_dim=variational_config.latent_dim,
        hidden_sizes=variational_config.velocity_hidden_sizes,
    ).to(device)

    encoder = VariationalForwardEncoder(
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

            mean, logvar = encoder(batch.x0, batch.x1)
            std = torch.exp(0.5 * logvar)
            latent_sample = mean + std * latent_noise
            latent_time_derivative = torch.zeros_like(latent_sample)

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

    return VariationalForwardMeanModifiedExperimentArtifacts(
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


class _VariationalForwardMeanTrajectoryWrapper(torch.nn.Module):
    """Wrapper that reuses the variational velocity model for forward mean sampling."""

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


def compute_variational_forward_mean_trajectories(
    model: VariationalVelocityMLP,
    x0: torch.Tensor,
    device: torch.device,
    integrator_config: IntegratorConfig,
    variational_config: VariationalForwardMeanFlowConfig,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate variational forward mean flow trajectories by sampling the latent code from the prior."""
    model.eval()
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
    wrapper = _VariationalForwardMeanTrajectoryWrapper(model, z)
    with torch.no_grad():
        trajectory, times = integrator.integrate(wrapper, x0_device, device)
    return trajectory, times


__all__ = [
    "ExperimentArtifacts",
    "VariationalExperimentArtifacts",
    "VariationalMeanFlowExperimentArtifacts",
    "VariationalMeanFlowModifiedExperimentArtifacts",
    "VariationalForwardMeanExperimentArtifacts",
    "VariationalForwardMeanModifiedExperimentArtifacts",
    "MeanFlowExperimentArtifacts",
    "VariationalTrainingHistory",
    "train_flow_matching",
    "train_mean_flow_matching",
    "train_variational_mean_flow_matching",
    "train_variational_mean_flow_modified_matching",
    "train_rectified_flow",
    "train_variational_flow_matching",
    "train_variational_forward_mean_flow_matching",
    "train_variational_forward_mean_flow_modified_matching",
    "generate_ground_truth",
    "generate_cheat_ground_truth",
    "compute_model_trajectories",
    "compute_mean_flow_trajectories",
    "compute_variational_mean_flow_trajectories",
    "compute_variational_trajectories",
    "compute_variational_forward_mean_trajectories",
]
