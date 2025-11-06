from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flowviz.config import (
    DATASET_CONFIGS,
    DatasetConfig,
    IntegratorConfig,
    RectifiedFlowConfig,
    TrainingConfig,
    VariationalFlowConfig,
    VariationalMeanFlowConfig,
    VariationalModifiedMeanFlowConfig,
)
from flowviz.pipelines.flow_matching import (
    compute_model_trajectories,
    compute_variational_trajectories,
    compute_variational_mean_trajectories,
    compute_variational_modified_mean_trajectories,
    generate_ground_truth,
    train_flow_matching,
    train_rectified_flow,
    train_variational_flow_matching,
    train_variational_mean_flow_matching,
    train_variational_modified_mean_flow_matching,
)
from flowviz.seed import create_generator, seed_all
from flowviz.visualization.plotting import (
    create_1d_trajectory_figure,
    create_2d_trajectory_figure,
    save_figure,
)


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
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "1d_default",
            "2d_default",
            "1d_wide_source_narrow_target",
            "2d_hexagonal_target",
            "2d_eight_gaussians_to_moons",
        ],
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset configurations to train and visualize",
    )
    parser.add_argument("--variational-latent-dim", type=int, default=8, help="Latent dimensionality for VFM")
    parser.add_argument("--variational-kl-weight", type=float, default=1.0, help="KL divergence weight for VFM")
    parser.add_argument(
        "--variational-matching-weight",
        type=float,
        default=1.0,
        help="Weight for flow-matching loss term in VFM",
    )
    parser.add_argument(
        "--variational-mean-latent-dim",
        type=int,
        default=None,
        help="Latent dimensionality for VMF (defaults to the VFM latent dim if omitted)",
    )
    parser.add_argument(
        "--variational-mean-kl-weight",
        type=float,
        default=None,
        help="KL divergence weight for VMF (defaults to the VFM weight if omitted)",
    )
    parser.add_argument(
        "--variational-mean-matching-weight",
        type=float,
        default=None,
        help="Matching loss weight for VMF (defaults to the VFM weight if omitted)",
    )
    parser.add_argument(
        "--variational-modified-latent-dim",
        type=int,
        default=None,
        help="Latent dimensionality for modified VMF (defaults to the VMF latent dim if omitted)",
    )
    parser.add_argument(
        "--variational-modified-kl-weight",
        type=float,
        default=None,
        help="KL divergence weight for modified VMF (defaults to the VMF weight if omitted)",
    )
    parser.add_argument(
        "--variational-modified-matching-weight",
        type=float,
        default=None,
        help="Matching loss weight for modified VMF (defaults to the VMF weight if omitted)",
    )
    parser.add_argument(
        "--variational-modified-p-mean-t",
        type=float,
        default=-1.2,
        help="Mean of the logit-normal prior used to sample t in modified VMF",
    )
    parser.add_argument(
        "--variational-modified-p-std-t",
        type=float,
        default=1.2,
        help="Standard deviation of the logit-normal prior used to sample t in modified VMF",
    )
    parser.add_argument(
        "--variational-modified-p-mean-r",
        type=float,
        default=-1.2,
        help="Mean of the logit-normal prior used to sample r in modified VMF",
    )
    parser.add_argument(
        "--variational-modified-p-std-r",
        type=float,
        default=1.2,
        help="Standard deviation of the logit-normal prior used to sample r in modified VMF",
    )
    parser.add_argument(
        "--variational-modified-ratio",
        type=float,
        default=0.5,
        help="Probability of sampling distinct timesteps in modified VMF",
    )
    parser.add_argument(
        "--show-ground-truth",
        type=bool,
        default=False,
        help="Whether to overlay ground-truth trajectories on model visualizations (true or false)",
    )
    parser.add_argument(
        "--max-display-1d",
        type=int,
        default=64,
        help="Maximum number of trajectories to display for 1D visualizations",
    )
    parser.add_argument(
        "--max-display-2d",
        type=int,
        default=128,
        help="Maximum number of trajectories to display for 2D visualizations",
    )
    args = parser.parse_args()

    if args.max_display_1d <= 0:
        parser.error("--max-display-1d must be a positive integer")
    if args.max_display_2d <= 0:
        parser.error("--max-display-2d must be a positive integer")
    if not 0.0 <= args.variational_modified_ratio <= 1.0:
        parser.error("--variational-modified-ratio must lie in [0, 1]")

    return args


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
    variational_mean_config = VariationalMeanFlowConfig(
        latent_dim=(
            args.variational_mean_latent_dim
            if args.variational_mean_latent_dim is not None
            else args.variational_latent_dim
        ),
        kl_weight=(
            args.variational_mean_kl_weight
            if args.variational_mean_kl_weight is not None
            else args.variational_kl_weight
        ),
        matching_weight=(
            args.variational_mean_matching_weight
            if args.variational_mean_matching_weight is not None
            else args.variational_matching_weight
        ),
    )
    variational_modified_config = VariationalModifiedMeanFlowConfig(
        latent_dim=(
            args.variational_modified_latent_dim
            if args.variational_modified_latent_dim is not None
            else variational_mean_config.latent_dim
        ),
        kl_weight=(
            args.variational_modified_kl_weight
            if args.variational_modified_kl_weight is not None
            else variational_mean_config.kl_weight
        ),
        matching_weight=(
            args.variational_modified_matching_weight
            if args.variational_modified_matching_weight is not None
            else variational_mean_config.matching_weight
        ),
        p_mean_t=args.variational_modified_p_mean_t,
        p_std_t=args.variational_modified_p_std_t,
        p_mean_r=args.variational_modified_p_mean_r,
        p_std_r=args.variational_modified_p_std_r,
        ratio=args.variational_modified_ratio,
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_configs: list[DatasetConfig] = [DATASET_CONFIGS[name] for name in args.datasets]

    for dataset_config in selected_configs:
        dataset = dataset_config.create_dataset(args.seed)
        key = dataset_config.name

        print(f"Training standard flow matching for {dataset_config.label}...")
        dataset.reset_rng(args.seed)
        fm_artifacts = train_flow_matching(dataset, training_config)

        dataset.reset_rng(args.seed)
        eval_batch = dataset.sample_pairs(args.eval_samples, device)
        gt_trajectory, times = generate_ground_truth(eval_batch.x0, eval_batch.x1, integrator_config.num_steps)
        predicted_trajectory, _ = compute_model_trajectories(
            fm_artifacts.model, eval_batch.x0, device, integrator_config
        )

        print(f"Training rectified flow for {dataset_config.label}...")
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

        print(f"Training variational flow matching for {dataset_config.label}...")
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

        print(f"Training variational mean flow matching for {dataset_config.label}...")
        dataset.reset_rng(args.seed)
        variational_mean_artifacts = train_variational_mean_flow_matching(
            dataset,
            training_config,
            variational_mean_config,
        )
        variational_mean_predicted, variational_mean_times = compute_variational_mean_trajectories(
            variational_mean_artifacts.velocity_model,
            eval_batch.x0,
            device,
            integrator_config,
            variational_mean_config,
            generator=inference_generator,
        )

        print(f"Training variational modified mean flow matching for {dataset_config.label}...")
        dataset.reset_rng(args.seed)
        variational_modified_artifacts = train_variational_modified_mean_flow_matching(
            dataset,
            training_config,
            variational_modified_config,
        )
        variational_modified_predicted, variational_modified_times = (
            compute_variational_modified_mean_trajectories(
                variational_modified_artifacts.velocity_model,
                eval_batch.x0,
                device,
                integrator_config,
                variational_modified_config,
                generator=inference_generator,
            )
        )

        if dataset.dim == 1:
            figures = {
                "ground_truth": create_1d_trajectory_figure(
                    times, gt_trajectory, "Ground Truth (1D)", max_display=args.max_display_1d
                ),
                "flow_matching": create_1d_trajectory_figure(
                    times,
                    predicted_trajectory,
                    "Flow Matching (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    max_display=args.max_display_1d,
                    show_reference=args.show_ground_truth,
                ),
                "rectified_flow": create_1d_trajectory_figure(
                    times,
                    rectified_predicted,
                    "Rectified Flow (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    max_display=args.max_display_1d,
                    show_reference=args.show_ground_truth,
                ),
                "variational_flow": create_1d_trajectory_figure(
                    variational_times,
                    variational_predicted,
                    "Variational Flow Matching (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    max_display=args.max_display_1d,
                    show_reference=args.show_ground_truth,
                ),
                "variational_mean_flow": create_1d_trajectory_figure(
                    variational_mean_times,
                    variational_mean_predicted,
                    "Variational Mean Flow (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    max_display=args.max_display_1d,
                    show_reference=args.show_ground_truth,
                ),
                "variational_modified_mean_flow": create_1d_trajectory_figure(
                    variational_modified_times,
                    variational_modified_predicted,
                    "Variational Modified Mean Flow (1D)",
                    reference=gt_trajectory,
                    reference_times=times,
                    max_display=args.max_display_1d,
                    show_reference=args.show_ground_truth,
                ),
            }
        else:
            figures = {
                "ground_truth": create_2d_trajectory_figure(
                    gt_trajectory, "Ground Truth (2D)", max_display=args.max_display_2d
                ),
                "flow_matching": create_2d_trajectory_figure(
                    predicted_trajectory,
                    "Flow Matching (2D)",
                    reference=gt_trajectory,
                    max_display=args.max_display_2d,
                    show_reference=args.show_ground_truth,
                ),
                "rectified_flow": create_2d_trajectory_figure(
                    rectified_predicted,
                    "Rectified Flow (2D)",
                    reference=gt_trajectory,
                    max_display=args.max_display_2d,
                    show_reference=args.show_ground_truth,
                ),
                "variational_flow": create_2d_trajectory_figure(
                    variational_predicted,
                    "Variational Flow Matching (2D)",
                    reference=gt_trajectory,
                    max_display=args.max_display_2d,
                    show_reference=args.show_ground_truth,
                ),
                "variational_mean_flow": create_2d_trajectory_figure(
                    variational_mean_predicted,
                    "Variational Mean Flow (2D)",
                    reference=gt_trajectory,
                    max_display=args.max_display_2d,
                    show_reference=args.show_ground_truth,
                ),
                "variational_modified_mean_flow": create_2d_trajectory_figure(
                    variational_modified_predicted,
                    "Variational Modified Mean Flow (2D)",
                    reference=gt_trajectory,
                    max_display=args.max_display_2d,
                    show_reference=args.show_ground_truth,
                ),
            }

        for name, fig in figures.items():
            filename = output_dir / f"{key}_{name}.png"
            save_figure(fig, filename)
            print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    main()
