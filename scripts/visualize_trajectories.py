from __future__ import annotations

import argparse
from pathlib import Path

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
from flowviz.seed import create_generator, seed_all
from flowviz.visualization.plotting import (
    create_1d_trajectory_figure,
    create_2d_trajectory_figure,
    save_figure,
)


def _str2bool(value: str) -> bool:
    truthy = {"yes", "true", "t", "1", "y"}
    falsy = {"no", "false", "f", "0", "n"}
    lower = value.lower()
    if lower in truthy:
        return True
    if lower in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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
    parser.add_argument("--eval-samples", type=int, default=1024, help="Evaluation samples for plotting")
    parser.add_argument("--variational-latent-dim", type=int, default=8, help="Latent dimensionality for VFM")
    parser.add_argument("--variational-kl-weight", type=float, default=1.0, help="KL divergence weight for VFM")
    parser.add_argument(
        "--variational-matching-weight",
        type=float,
        default=1.0,
        help="Weight for flow-matching loss term in VFM",
    )
    parser.add_argument(
        "--show-ground-truth",
        type=_str2bool,
        default=True,
        help="Whether to overlay ground-truth trajectories on model visualizations (true or false)",
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
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "1d": GaussianMixture1D(seed=args.seed),
        "2d": GaussianMixture2D(seed=args.seed),
    }

    for key, dataset in datasets.items():
        print(f"Training standard flow matching for {key} dataset...")
        dataset.reset_rng(args.seed)
        fm_artifacts = train_flow_matching(dataset, training_config)

        dataset.reset_rng(args.seed)
        eval_batch = dataset.sample_pairs(args.eval_samples, device)
        gt_trajectory, times = generate_ground_truth(eval_batch.x0, eval_batch.x1, integrator_config.num_steps)
        predicted_trajectory, _ = compute_model_trajectories(
            fm_artifacts.model, eval_batch.x0, device, integrator_config
        )

        print(f"Training rectified flow for {key} dataset...")
        dataset.reset_rng(args.seed)
        rectified_artifacts, _ = train_rectified_flow(
            fm_artifacts.model,
            dataset,
            training_config,
            integrator_config,
            rectified_config,
        )

        rectified_predicted, _ = compute_model_trajectories(
            rectified_artifacts.model, eval_batch.x0, device, integrator_config
        )

        print(f"Training variational flow matching for {key} dataset...")
        dataset.reset_rng(args.seed)
        variational_artifacts = train_variational_flow_matching(dataset, training_config, variational_config)
        inference_generator = create_generator(args.seed, device=device.type)
        variational_predicted, variational_times = compute_variational_trajectories(
            variational_artifacts.velocity_model,
            eval_batch.x0,
            device,
            integrator_config,
            variational_config,
            generator=inference_generator,
        )

        if dataset.dim == 1:
            figures = {
                "ground_truth": create_1d_trajectory_figure(times, gt_trajectory, "Ground Truth (1D)"),
                "flow_matching": create_1d_trajectory_figure(
                    times,
                    predicted_trajectory,
                    "Flow Matching (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    show_reference=args.show_ground_truth,
                ),
                "rectified_flow": create_1d_trajectory_figure(
                    times,
                    rectified_predicted,
                    "Rectified Flow (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    show_reference=args.show_ground_truth,
                ),
                "variational_flow": create_1d_trajectory_figure(
                    variational_times,
                    variational_predicted,
                    "Variational Flow Matching (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    show_reference=args.show_ground_truth,
                ),
            }
        else:
            figures = {
                "ground_truth": create_2d_trajectory_figure(gt_trajectory, "Ground Truth (2D)"),
                "flow_matching": create_2d_trajectory_figure(
                    predicted_trajectory,
                    "Flow Matching (2D)",
                    reference=gt_trajectory,
                    show_reference=args.show_ground_truth,
                ),
                "rectified_flow": create_2d_trajectory_figure(
                    rectified_predicted,
                    "Rectified Flow (2D)",
                    reference=gt_trajectory,
                    show_reference=args.show_ground_truth,
                ),
                "variational_flow": create_2d_trajectory_figure(
                    variational_predicted,
                    "Variational Flow Matching (2D)",
                    reference=gt_trajectory,
                    show_reference=args.show_ground_truth,
                ),
            }

        for name, fig in figures.items():
            filename = output_dir / f"{key}_{name}.png"
            save_figure(fig, filename)
            print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    main()
