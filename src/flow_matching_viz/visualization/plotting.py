"""Plotting utilities for Flow Matching trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..data.synthetic import GaussianMixture1D, TwoMoons2D


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def plot_1d_trajectories(
    times: torch.Tensor,
    trajectories: torch.Tensor,
    dataset: GaussianMixture1D,
    output_path: str | Path,
    max_curves: int = 32,
) -> Path:
    """Create a time-vs-state plot and histogram for the one dimensional case."""

    output_path = _to_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    times_np = times.numpy()
    traj_np = trajectories.squeeze(-1).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    selection = traj_np[: max_curves]
    for curve in selection:
        axes[0].plot(times_np, curve, alpha=0.6)
    axes[0].set_title("1D Flow Matching trajectories")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("State")

    target_samples = dataset.sample(10_000).numpy().squeeze(-1)
    generated_samples = trajectories[:, -1].numpy().squeeze(-1)
    axes[1].hist(target_samples, bins=60, density=True, alpha=0.6, label="Target")
    axes[1].hist(generated_samples, bins=60, density=True, alpha=0.6, label="Generated")
    axes[1].set_title("Distribution comparison")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_2d_trajectories(
    times: torch.Tensor,
    trajectories: torch.Tensor,
    dataset: TwoMoons2D,
    output_path: str | Path,
    max_curves: int = 200,
) -> Path:
    """Plot planar trajectories along with the target distribution."""

    output_path = _to_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traj_np = trajectories.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectories
    subset = traj_np[:max_curves]
    for curve in subset:
        axes[0].plot(curve[:, 0], curve[:, 1], alpha=0.5)
    axes[0].set_title("2D Flow Matching trajectories")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    target_samples = dataset.sample(10_000).numpy()
    axes[1].scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        s=4,
        alpha=0.5,
        label="Target",
    )
    generated_samples = trajectories[:, -1].numpy()
    axes[1].scatter(
        generated_samples[:, 0],
        generated_samples[:, 1],
        s=4,
        alpha=0.5,
        label="Generated",
    )
    axes[1].set_title("Distribution comparison")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
