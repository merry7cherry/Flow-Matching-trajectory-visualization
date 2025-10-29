"""Train Flow Matching variants on synthetic data and generate visualisations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from flow_matching_trajectory import SeedConfig, set_global_seed
from flow_matching_trajectory.data.synthetic import (
    RectifiedPairDataset,
    SyntheticPairDataset,
    build_1d_mixture_dataset,
    build_2d_spiral_dataset,
)
from flow_matching_trajectory.flows.linear import LinearFlowMatching
from flow_matching_trajectory.flows.rectified import RectifiedFlowMatching, generate_rectified_pairs
from flow_matching_trajectory.flows.variational import VariationalFlowMatching
from flow_matching_trajectory.training.trainer import FlowMatchingTrainer, TrainerConfig
from flow_matching_trajectory.utils.integration import euler_integrate
from flow_matching_trajectory.visualization.plotter import plot_1d_trajectories, plot_2d_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3_000, help="Training steps for each variant")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--integration-steps", type=int, default=50, help="Euler steps for simulation")
    parser.add_argument("--num-trajectories", type=int, default=32)
    return parser.parse_args()


def build_datasets(device: torch.device) -> Dict[str, SyntheticPairDataset]:
    return {
        "1d": build_1d_mixture_dataset(device),
        "2d": build_2d_spiral_dataset(device),
    }


@torch.no_grad()
def simulate_variant(
    variant,
    x0: torch.Tensor,
    times: torch.Tensor,
    context: torch.Tensor | None = None,
) -> torch.Tensor:
    variant.eval()
    batch_size = x0.shape[0]
    if context is None:
        context = variant.sample_inference_context(batch_size, x0.device)
    if context is not None and context.shape[0] != batch_size:
        raise ValueError("Context batch size mismatch")

    def velocity_fn(state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return variant.predict_velocity(state, t, context)

    trajectory, _ = euler_integrate(velocity_fn, x0, times)
    return trajectory


def compute_ground_truth(batch, times: torch.Tensor) -> torch.Tensor:
    x0 = batch.x0
    x1 = batch.x1
    t = times.view(-1, 1, 1)
    return (1.0 - t) * x0.unsqueeze(0) + t * x1.unsqueeze(0)


def train_variant(variant, dataset: SyntheticPairDataset, config: TrainerConfig) -> None:
    trainer = FlowMatchingTrainer(variant.to(config.device), dataset, config)
    trainer.train()


def run_pipeline(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_global_seed(SeedConfig(seed=args.seed))
    datasets = build_datasets(device)
    times = torch.linspace(0.0, 1.0, args.integration_steps + 1, device=device)

    for key, dataset in datasets.items():
        dim = dataset.sample(1).x0.shape[1]
        output_root = args.output_dir / key
        output_root.mkdir(parents=True, exist_ok=True)

        # Standard Flow Matching
        linear_variant = LinearFlowMatching(dim=dim)
        train_variant(
            linear_variant,
            dataset,
            TrainerConfig(
                batch_size=args.batch_size,
                num_steps=args.steps,
                device=device,
            ),
        )

        eval_batch = dataset.sample(args.num_trajectories)
        gt_traj = compute_ground_truth(eval_batch, times)
        learned_traj = simulate_variant(linear_variant, eval_batch.x0, times)

        if dim == 1:
            plot_1d_trajectories(
                times,
                gt_traj,
                learned_traj,
                title=f"Standard Flow Matching ({key})",
                output_path=output_root / "linear_flow_matching.png",
            )
        else:
            plot_2d_trajectories(
                gt_traj,
                learned_traj,
                title=f"Standard Flow Matching ({key})",
                output_path=output_root / "linear_flow_matching.png",
            )

        # Rectified Flow
        rectified_pairs = generate_rectified_pairs(
            linear_variant,
            dataset,
            num_samples=max(args.batch_size, args.num_trajectories * 8),
            integration_steps=args.integration_steps,
            device=device,
        )
        rectified_dataset = RectifiedPairDataset(rectified_pairs, device)
        rectified_variant = RectifiedFlowMatching(dim=dim)
        train_variant(
            rectified_variant,
            rectified_dataset,
            TrainerConfig(
                batch_size=args.batch_size,
                num_steps=args.steps // 2,
                device=device,
            ),
        )

        rectified_eval = rectified_dataset.sample(args.num_trajectories)
        rectified_gt = compute_ground_truth(rectified_eval, times)
        rectified_learned = simulate_variant(rectified_variant, rectified_eval.x0, times)

        if dim == 1:
            plot_1d_trajectories(
                times,
                rectified_gt,
                rectified_learned,
                title=f"Rectified Flow ({key})",
                output_path=output_root / "rectified_flow.png",
            )
        else:
            plot_2d_trajectories(
                rectified_gt,
                rectified_learned,
                title=f"Rectified Flow ({key})",
                output_path=output_root / "rectified_flow.png",
            )

        # Variational Flow Matching
        vfm_variant = VariationalFlowMatching(dim=dim)
        train_variant(
            vfm_variant,
            dataset,
            TrainerConfig(
                batch_size=args.batch_size,
                num_steps=args.steps,
                device=device,
            ),
        )

        vfm_eval = dataset.sample(args.num_trajectories)
        vfm_gt = compute_ground_truth(vfm_eval, times)
        context = vfm_variant.sample_inference_context(args.num_trajectories, device)
        vfm_learned = simulate_variant(vfm_variant, vfm_eval.x0, times, context=context)

        if dim == 1:
            plot_1d_trajectories(
                times,
                vfm_gt,
                vfm_learned,
                title=f"Variational Flow Matching ({key})",
                output_path=output_root / "variational_flow_matching.png",
            )
        else:
            plot_2d_trajectories(
                vfm_gt,
                vfm_learned,
                title=f"Variational Flow Matching ({key})",
                output_path=output_root / "variational_flow_matching.png",
            )


if __name__ == "__main__":
    run_pipeline(parse_args())
