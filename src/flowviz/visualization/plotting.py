from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _select_indices(num_trajectories: int, max_display: int) -> torch.Tensor:
    if num_trajectories <= max_display:
        return torch.arange(num_trajectories)
    step = max(1, num_trajectories // max_display)
    return torch.arange(0, num_trajectories, step)[:max_display]


def plot_1d_trajectories(
    ax: plt.Axes,
    times: torch.Tensor,
    ground_truth: torch.Tensor,
    predicted: torch.Tensor,
    title: str,
    max_display: int = 8,
) -> None:
    indices = _select_indices(ground_truth.shape[1], max_display)
    times_np = times.cpu().numpy()
    for idx in indices:
        gt = ground_truth[:, idx, 0].cpu().numpy()
        pred = predicted[:, idx, 0].cpu().numpy()
        ax.plot(times_np, gt, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.plot(times_np, pred, linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)


def plot_2d_trajectories(
    ax: plt.Axes,
    ground_truth: torch.Tensor,
    predicted: torch.Tensor,
    title: str,
    max_display: int = 12,
) -> None:
    indices = _select_indices(ground_truth.shape[1], max_display)
    for idx in indices:
        gt = ground_truth[:, idx].cpu().numpy()
        pred = predicted[:, idx].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], linestyle="--", linewidth=1.2, alpha=0.8)
        ax.plot(pred[:, 0], pred[:, 1], linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


__all__ = [
    "plot_1d_trajectories",
    "plot_2d_trajectories",
    "save_figure",
]
