"""Plotting utilities for Flow Matching trajectory visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch


def _prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_1d_trajectories(
    times: torch.Tensor,
    ground_truth: torch.Tensor,
    learned: torch.Tensor,
    title: str,
    output_path: Path,
    legend_labels: Optional[Iterable[str]] = None,
) -> None:
    _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4))
    for traj in ground_truth.transpose(0, 1):
        plt.plot(times.cpu().numpy(), traj.squeeze(-1).cpu().numpy(), color="tab:blue", alpha=0.4)
    for traj in learned.transpose(0, 1):
        plt.plot(times.cpu().numpy(), traj.squeeze(-1).cpu().numpy(), color="tab:orange", alpha=0.6)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    labels = list(legend_labels) if legend_labels is not None else ["ground truth", "learned"]
    plt.plot([], [], color="tab:blue", label=labels[0])
    plt.plot([], [], color="tab:orange", label=labels[1] if len(labels) > 1 else "learned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_2d_trajectories(
    ground_truth: torch.Tensor,
    learned: torch.Tensor,
    title: str,
    output_path: Path,
    times: Optional[torch.Tensor] = None,
) -> None:
    _prepare_output_path(output_path)
    plt.figure(figsize=(6, 6))
    for traj in ground_truth.transpose(0, 1):
        plt.plot(traj[:, 0].cpu().numpy(), traj[:, 1].cpu().numpy(), color="tab:blue", alpha=0.4)
    for traj in learned.transpose(0, 1):
        plt.plot(traj[:, 0].cpu().numpy(), traj[:, 1].cpu().numpy(), color="tab:orange", alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.plot([], [], color="tab:blue", label="ground truth")
    plt.plot([], [], color="tab:orange", label="learned")
    plt.legend()
    if times is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
