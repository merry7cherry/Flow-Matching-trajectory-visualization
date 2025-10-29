from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from flowviz.config import (
    IntegratorConfig,
    RectifiedFlowConfig,
    TrainingConfig,
    VariationalFlowConfig,
)
from flowviz.data.synthetic import GaussianMixture1D, GaussianMixture2D
from flowviz.pipelines.flow_matching import (
    compute_model_trajectories,
    compute_variational_trajectories,
    generate_ground_truth,
    train_flow_matching,
    train_rectified_flow,
    train_variational_flow_matching,
)
from flowviz.seed import seed_all
from flowviz.visualization.plotting import plot_1d_trajectories, plot_2d_trajectories, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow Matching trajectory visualization")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory for figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Gradient steps per epoch")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--integrator-steps", type=int, default=60, help="Number of ODE steps for Euler integration")
    parser.add_argument("--rectified-samples", type=int, default=6000, help="Samples for rectified dataset")
    parser.add_argument("--rectified-batch", type=int, default=512, help="Batch size during rectified dataset generation")
    parser.add_argument("--eval-samples", type=int, default=256, help="Evaluation samples for plotting")
    parser.add_argument("--variational-latent-dim", type=int, default=8, help="Latent dimensionality for VFM")
    parser.add_argument("--variational-kl-weight", type=float, default=1.0, help="KL divergence weight for VFM")
    parser.add_argument(
        "--variational-matching-weight",
        type=float,
        default=1.0,
        help="Weight for ||v||^2-Δ^X term in VFM",
    )
    parser.add_argument(
        "--variational-reconstruction-weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction-style penalty in VFM",
    )
    return parser.parse_args()


def prepare_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.lr,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)
    training_config = prepare_training_config(args)
    integrator_config = IntegratorConfig(num_steps=args.integrator_steps)
    rectified_config = RectifiedFlowConfig(num_samples=args.rectified_samples, batch_size=args.rectified_batch)
    variational_config = VariationalFlowConfig(
        latent_dim=args.variational_latent_dim,
        kl_weight=args.variational_kl_weight,
        matching_weight=args.variational_matching_weight,
        reconstruction_weight=args.variational_reconstruction_weight,
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "1d": GaussianMixture1D(seed=args.seed),
        "2d": GaussianMixture2D(seed=args.seed),
    }

    for key, dataset in datasets.items():
        print(f"Training standard flow matching for {key} dataset...")
        fm_artifacts = train_flow_matching(dataset, training_config)

        eval_batch = dataset.sample_pairs(args.eval_samples, device)
        gt_trajectory, times = generate_ground_truth(eval_batch.x0, eval_batch.x1, integrator_config.num_steps)
        predicted_trajectory, _ = compute_model_trajectories(
            fm_artifacts.model, eval_batch.x0, device, integrator_config
        )

        print(f"Training rectified flow for {key} dataset...")
        rectified_artifacts, builder = train_rectified_flow(
            fm_artifacts.model,
            dataset,
            training_config,
            integrator_config,
            rectified_config,
        )

        with torch.no_grad():
            rectified_targets, _ = builder.build_ground_truth(fm_artifacts.model, eval_batch.x0)
        rectified_gt_trajectory, _ = generate_ground_truth(eval_batch.x0, rectified_targets, integrator_config.num_steps)
        rectified_predicted, _ = compute_model_trajectories(
            rectified_artifacts.model, eval_batch.x0, device, integrator_config
        )

        print(f"Training variational flow matching for {key} dataset...")
        variational_artifacts = train_variational_flow_matching(dataset, training_config, variational_config)
        variational_predicted, variational_times = compute_variational_trajectories(
            variational_artifacts.velocity_model,
            eval_batch.x0,
            device,
            integrator_config,
            variational_config,
        )

        if dataset.dim == 1:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            plot_1d_trajectories(axes[0], times, gt_trajectory, predicted_trajectory, "Flow Matching (1D)")
            plot_1d_trajectories(
                axes[1], times, rectified_gt_trajectory, rectified_predicted, "Rectified Flow (1D)"
            )
            plot_1d_trajectories(
                axes[2],
                variational_times,
                gt_trajectory,
                variational_predicted,
                "Variational Flow Matching (1D)",
            )
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            plot_2d_trajectories(axes[0], gt_trajectory, predicted_trajectory, "Flow Matching (2D)")
            plot_2d_trajectories(
                axes[1], rectified_gt_trajectory, rectified_predicted, "Rectified Flow (2D)"
            )
            plot_2d_trajectories(
                axes[2], gt_trajectory, variational_predicted, "Variational Flow Matching (2D)"
            )

        filename = output_dir / f"{key}_trajectories.png"
        save_figure(fig, filename)
        print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    main()
