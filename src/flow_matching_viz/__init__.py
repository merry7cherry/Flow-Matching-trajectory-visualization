"""Top-level package for Flow Matching trajectory visualization utilities."""

from .flows.standard import StandardFlowMatching
from .models.mlp import TimeConditionedMLP
from .data.synthetic import GaussianMixture1D, TwoMoons2D
from .visualization.trajectory import sample_trajectories
from .visualization.plotting import plot_1d_trajectories, plot_2d_trajectories

__all__ = [
    "StandardFlowMatching",
    "TimeConditionedMLP",
    "GaussianMixture1D",
    "TwoMoons2D",
    "sample_trajectories",
    "plot_1d_trajectories",
    "plot_2d_trajectories",
]
