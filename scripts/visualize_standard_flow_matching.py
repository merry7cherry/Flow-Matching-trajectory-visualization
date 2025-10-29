"""Command line entry point for Flow Matching trajectory visualisation."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from flow_matching_viz import (
    GaussianMixture1D,
    StandardFlowMatching,
    TimeConditionedMLP,
    TwoMoons2D,
    plot_1d_trajectories,
    plot_2d_trajectories,
    sample_trajectories,
)
from flow_matching_viz.training.trainer import FlowMatchingTrainer, TrainingConfig
from flow_matching_viz.utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Torch device to run on.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--steps-1d", type=int, default=3000)
    parser.add_argument("--steps-2d", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--trajectory-samples", type=int, default=1024)
    parser.add_argument("--trajectory-steps", type=int, default=200)
    return parser


def train_model(
    dataset,
    data_dim: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
) -> StandardFlowMatching:
    model = TimeConditionedMLP(input_dim=data_dim + 1, hidden_layers=(128, 128, 128))
    flow = StandardFlowMatching(model, data_dim=data_dim, device=device)
    optimizer = torch.optim.Adam(flow.velocity_model.parameters(), lr=lr)
    config = TrainingConfig(batch_size=batch_size, steps=steps, log_every=max(steps // 10, 1))
    trainer = FlowMatchingTrainer(
        flow,
        dataset_sampler=dataset.sample,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    losses = trainer.train()
    print(f"Training finished. Final logged loss: {losses[-1]:.4f}")
    return flow


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Training 1D Flow Matching model...")
    dataset_1d = GaussianMixture1D()
    flow_1d = train_model(
        dataset_1d,
        data_dim=1,
        device=device,
        steps=args.steps_1d,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    times_1d, trajectories_1d = sample_trajectories(
        flow_1d, batch_size=args.trajectory_samples, num_steps=args.trajectory_steps
    )
    output_1d = args.output_dir / "flow_matching_1d.png"
    plot_1d_trajectories(times_1d, trajectories_1d, dataset_1d, output_1d)
    print(f"1D visualisation saved to {output_1d}")

    print("Training 2D Flow Matching model...")
    dataset_2d = TwoMoons2D()
    flow_2d = train_model(
        dataset_2d,
        data_dim=2,
        device=device,
        steps=args.steps_2d,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    times_2d, trajectories_2d = sample_trajectories(
        flow_2d, batch_size=args.trajectory_samples, num_steps=args.trajectory_steps
    )
    output_2d = args.output_dir / "flow_matching_2d.png"
    plot_2d_trajectories(times_2d, trajectories_2d, dataset_2d, output_2d)
    print(f"2D visualisation saved to {output_2d}")


if __name__ == "__main__":
    main(build_parser().parse_args())
