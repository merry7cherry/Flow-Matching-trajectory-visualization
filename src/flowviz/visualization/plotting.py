from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch

SOURCE_COLOR = "#1f77b4"
TARGET_COLOR = "#d62728"
REFERENCE_COLOR = "#7f7f7f"


def _select_indices(num_trajectories: int, max_display: int) -> torch.Tensor:
    if num_trajectories <= max_display:
        return torch.arange(num_trajectories)
    step = max(1, num_trajectories // max_display)
    return torch.arange(0, num_trajectories, step)[:max_display]


def _maybe_numpy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def _plot_endpoint_markers(
    ax: plt.Axes,
    start: Iterable[float],
    end: Iterable[float],
    is_reference: bool,
    dim: int,
) -> None:
    marker = "x" if is_reference else "o"
    size = 40 if dim == 2 else 30
    ax.scatter(*start, color=SOURCE_COLOR, marker=marker, s=size, alpha=0.9)
    ax.scatter(*end, color=TARGET_COLOR, marker=marker, s=size, alpha=0.9)


def create_1d_trajectory_figure(
    times: torch.Tensor,
    trajectories: torch.Tensor,
    title: str,
    reference: torch.Tensor | None = None,
    reference_times: torch.Tensor | None = None,
    max_display: int = 16,
    show_reference: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    indices = _select_indices(trajectories.shape[1], max_display)

    times_np = _maybe_numpy(times).numpy()
    reference_times_np = _maybe_numpy(reference_times).numpy() if reference_times is not None else times_np

    reference_label_added = False
    prediction_label_added = False

    for idx in indices.tolist():
        traj = _maybe_numpy(trajectories[:, idx, 0]).numpy()
        ax.plot(
            times_np,
            traj,
            linewidth=1.5,
            alpha=0.9,
            label="Predicted" if not prediction_label_added else None,
        )
        prediction_label_added = True

        start = ([times_np[0]], [traj[0]])
        end = ([times_np[-1]], [traj[-1]])
        _plot_endpoint_markers(ax, start, end, is_reference=False, dim=1)

        if reference is not None and show_reference:
            ref_traj = _maybe_numpy(reference[:, idx, 0]).numpy()
            ax.plot(
                reference_times_np,
                ref_traj,
                linestyle="--",
                linewidth=1.2,
                color=REFERENCE_COLOR,
                alpha=0.9,
                label="Ground Truth" if not reference_label_added else None,
            )
            reference_label_added = True

            ref_start = ([reference_times_np[0]], [ref_traj[0]])
            ref_end = ([reference_times_np[-1]], [ref_traj[-1]])
            _plot_endpoint_markers(ax, ref_start, ref_end, is_reference=True, dim=1)

    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    if reference is not None and show_reference:
        ax.legend(loc="upper left")
    return fig


def create_2d_trajectory_figure(
    trajectories: torch.Tensor,
    title: str,
    reference: torch.Tensor | None = None,
    max_display: int = 24,
    show_reference: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    indices = _select_indices(trajectories.shape[1], max_display)

    reference_label_added = False
    prediction_label_added = False

    for idx in indices.tolist():
        traj = _maybe_numpy(trajectories[:, idx]).numpy()
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=1.5,
            alpha=0.9,
            label="Predicted" if not prediction_label_added else None,
        )
        prediction_label_added = True
        _plot_endpoint_markers(
            ax,
            (traj[0, 0], traj[0, 1]),
            (traj[-1, 0], traj[-1, 1]),
            is_reference=False,
            dim=2,
        )

        if reference is not None and show_reference:
            ref_traj = _maybe_numpy(reference[:, idx]).numpy()
            ax.plot(
                ref_traj[:, 0],
                ref_traj[:, 1],
                linestyle="--",
                linewidth=1.2,
                color=REFERENCE_COLOR,
                alpha=0.9,
                label="Ground Truth" if not reference_label_added else None,
            )
            reference_label_added = True
            _plot_endpoint_markers(
                ax,
                (ref_traj[0, 0], ref_traj[0, 1]),
                (ref_traj[-1, 0], ref_traj[-1, 1]),
                is_reference=True,
                dim=2,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    if reference is not None and show_reference:
        ax.legend(loc="upper left")
    return fig


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "create_1d_trajectory_figure",
    "create_2d_trajectory_figure",
    "save_figure",
]
