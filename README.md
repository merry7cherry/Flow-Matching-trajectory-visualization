# Flow Matching Trajectory Visualization

This project visualizes flow matching trajectories for 1D and 2D synthetic datasets. It provides a modular codebase that makes it straightforward to add new flow-matching variants, models, and datasets.

## Features

- **Standard Flow Matching** with linear interpolation.
- **Rectified Flow** training using the trajectories produced by the base flow-matching model.
- **Variational Flow Matching (VFM)** with a latent-conditioned velocity field trained jointly with a variational encoder.
- **Variational Mean Flow (VMF)** that augments the variational architecture with forward-mode derivatives to match the mean field objective.
- **Variational Modified Mean Flow (VMMF)** extending VMF with an auxiliary timestep sampler and time-conditioned networks.
- Modular interfaces for datasets, flow objectives, models, training, simulation, and visualization.
- Deterministic experiments via a fixed random seed.

## Project Layout

```
src/flowviz/
├── config.py                 # Dataclasses for experiment configuration
├── data/                     # Dataset interfaces and synthetic datasets
├── flows/                    # Flow-matching objectives and rectified flow utilities
├── models/                   # Neural network architectures
├── pipelines/                # High-level experiment orchestration
├── simulation/               # ODE integrators
├── training/                 # Training loops and utilities
└── visualization/            # Plotting helpers
scripts/
└── visualize_trajectories.py # Entry-point for reproducing the figures
```

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install torch matplotlib
   ```

2. Generate the visualizations:
   ```bash
   PYTHONPATH=src python scripts/visualize_trajectories.py --device cpu
   ```

   Images for the 1D and 2D experiments will be saved under the `outputs/` directory. The script trains the flow matching,
   rectified flow, variational flow matching, variational mean flow, and the modified variational mean flow baselines so you will
   find companion figures labelled `flow_matching`, `rectified_flow`, `variational_flow`, `variational_mean_flow`, and
   `variational_modified_mean_flow` for each dataset. Use the following optional flags to tune the variational flow experiments:

   - `--variational-latent-dim`: dimensionality of the latent code sampled from the VAE prior (default: 8)
   - `--variational-kl-weight`: weight for the KL divergence between encoder posterior and standard normal prior (default: 1.0)
   - `--variational-matching-weight`: weight for the flow-matching mean-squared error conditioned on the latent code (default: 1.0)
   - `--variational-mean-latent-dim`: optional latent dimensionality override for VMF (defaults to the VFM setting)
   - `--variational-mean-kl-weight`: optional KL divergence weight override for VMF (defaults to the VFM setting)
   - `--variational-mean-matching-weight`: optional matching loss weight override for VMF (defaults to the VFM setting)
   - `--variational-modified-latent-dim`: optional latent dimensionality override for the modified VMF (defaults to the VMF setting)
   - `--variational-modified-kl-weight`: optional KL divergence weight override for the modified VMF (defaults to the VMF setting)
   - `--variational-modified-matching-weight`: optional matching loss weight override for the modified VMF (defaults to the VMF setting)
   - `--variational-modified-p-mean-t`, `--variational-modified-p-std-t`: parameters of the logit-normal prior for sampling the main time `t`
   - `--variational-modified-p-mean-r`, `--variational-modified-p-std-r`: parameters of the logit-normal prior for sampling the auxiliary time `r`
   - `--variational-modified-ratio`: probability of sampling distinct `(t, r)` pairs (defaults to 0.5)

## Variational Mean Flow (VMF)

The variational mean flow shares the encoder and latent-conditioned velocity network with VFM, but optimizes the different target
introduced in the variational mean flow literature. The training loop in `flowviz.pipelines.flow_matching.train_variational_mean_flow_matching`
computes forward-mode derivatives with `torch.func.jvp` to build the detached flow-matching targets and records the loss terms needed for
analysis.

To visualize VMF trajectories, use `flowviz.pipelines.flow_matching.compute_variational_mean_trajectories`. The helper samples latent
codes from the standard normal prior (matching inference-time behaviour) and integrates the velocity field with the Euler integrator,
producing a tensor of states and their corresponding time stamps that can be plotted with the existing visualization utilities.

## Variational Modified Mean Flow (VMMF)

The variational modified mean flow extends VMF by sampling two correlated timesteps `(t, r)` from configurable logit-normal
distributions. Both the encoder and the latent-conditioned velocity model observe the auxiliary difference `h = t - r`, enabling the
forward-mode autodiff objective to incorporate additional temporal structure. The training loop in
`flowviz.pipelines.flow_matching.train_variational_modified_mean_flow_matching` mirrors the VMF implementation but swaps the
detached target for the modified objective `u_tgt = v - (t - r) * \partial_t u`.

Sampling leverages `flowviz.pipelines.flow_matching.compute_variational_modified_mean_trajectories`, which now performs the analytic
one-step update `z_0 = z_1 - u(z_1, t = 1, r = 0)` instead of numerically integrating an ODE. The helper evaluates the velocity model
once with `r = 0`, assembles a two-point trajectory (noise to generated sample), and returns timestamps aligned with the other
visualization utilities.

## Adding New Flow Variants

To extend the project with a new flow-matching variant:

1. Implement a new objective by subclassing `flowviz.flows.base.FlowMatchingObjective`.
2. Either reuse the provided training utilities (`flowviz.training.trainer.train_model` for deterministic objectives or
   `flowviz.pipelines.flow_matching.train_variational_flow_matching` as an example for latent-variable training), or create
   a custom training loop following the same interface.
3. Reuse the plotting utilities or create new ones under `flowviz.visualization` as needed.

This modular approach keeps the training, simulation, and visualization layers reusable across future experiments.
