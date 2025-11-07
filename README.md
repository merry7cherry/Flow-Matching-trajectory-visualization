# Flow Matching Trajectory Visualization

This project visualizes flow matching trajectories for 1D and 2D synthetic datasets. It provides a modular codebase that makes it straightforward to add new flow-matching variants, models, and datasets.

## Features

- **Standard Flow Matching** with linear interpolation.
- **Mean Flow** objective with an additional time parameter \(r\) and forward-mode differentiation during training.
- **Variational Mean Flow Modified** objective that augments the mean flow with a latent encoder conditioned on \((x_0, x_1)\).
- **Rectified Flow** training using the trajectories produced by the base flow-matching model.
- **Variational Flow Matching (VFM)** with a latent-conditioned velocity field trained jointly with a variational encoder.
- **Variational Forward Mean Flow (VFMF)** that augments the variational architecture with forward-mode derivatives to match the mean field objective.
- **Variational Forward Mean Flow Modified (VFMF-M)** that shares the latent-conditioned velocity network with VFMF while using a forward encoder conditioned only on \(x_0, x_1\).
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
   mean flow, variational mean flow modified, rectified flow, variational flow matching, variational forward mean flow, and variational forward mean flow modified baselines so you will find companion figures
   labelled `flow_matching`, `mean_flow`, `variational_mean_flow_modified`, `rectified_flow`, `variational_flow`, `variational_forward_mean_flow`, and `variational_forward_mean_flow_modified` for each dataset. Use the following optional
   flags to tune the variational flow experiments:

   - `--variational-latent-dim`: dimensionality of the latent code sampled from the VAE prior (default: 8)
   - `--variational-kl-weight`: weight for the KL divergence between encoder posterior and standard normal prior (default: 1.0)
   - `--variational-matching-weight`: weight for the flow-matching mean-squared error conditioned on the latent code (default: 1.0)
   - `--variational-forward-mean-latent-dim`: optional latent dimensionality override for VFMF (defaults to the VFM setting)
   - `--variational-forward-mean-kl-weight`: optional KL divergence weight override for VFMF (defaults to the VFM setting)
   - `--variational-forward-mean-matching-weight`: optional matching loss weight override for VFMF (defaults to the VFM setting)
   - `--variational-forward-mean-modified-latent-dim`: optional latent dimensionality override for VFMF-M (defaults to the VFM setting)
   - `--variational-forward-mean-modified-kl-weight`: optional KL divergence weight override for VFMF-M (defaults to the VFM setting)
   - `--variational-forward-mean-modified-matching-weight`: optional matching loss weight override for VFMF-M (defaults to the VFM setting)
   - `--mean-flow-steps`: number of uniform inference steps used when sampling the mean flow (default: 1)

## Variational Mean Flow Modified (VMFM)

The variational mean flow modified objective mirrors the mean flow training procedure while introducing a latent variable sampled from an encoder
that only depends on the endpoints \((x_0, x_1)\). Since the latent code is independent of the interpolation time, its forward
derivative vanishes and the resulting training loop reuses the mean flow Jacobian-vector products without additional latent
backpropagation terms. The implementation lives in `flowviz.pipelines.flow_matching.train_variational_mean_flow_modified_matching` and
records the same aggregate loss statistics used by the other variational objectives. During inference,
`flowviz.pipelines.flow_matching.compute_variational_mean_flow_modified_trajectories` samples latent codes from the standard normal prior
and integrates the latent-conditioned mean flow network with uniformly spaced Euler updates, matching the interface of the
deterministic mean flow helper.

## Mean Flow

The mean flow extends the standard flow-matching setup by sampling a pair of times \(t, r\) such that \(t \geq r\), computing the
offset \(h = t - r\), and predicting the velocity conditioned on \((t, h)\). The training procedure implemented in
`flowviz.pipelines.flow_matching.train_mean_flow_matching` uses `torch.func.jvp` to obtain forward-mode derivatives and constructs
the detached targets described by the mean flow objective, including the adaptive weighting term for stability. During inference,
`flowviz.pipelines.flow_matching.compute_mean_flow_trajectories` evaluates the network across uniformly spaced time steps between
0 and 1 (configurable via `--mean-flow-steps`) so you can trade off runtime for fidelity, while still supporting the one-step
generation path with the default setting.

## Variational Forward Mean Flow (VFMF)

The variational forward mean flow shares the encoder and latent-conditioned velocity network with VFM, but optimizes the different target
introduced in the variational forward mean flow literature. The training loop in `flowviz.pipelines.flow_matching.train_variational_forward_mean_flow_matching`
computes forward-mode derivatives with `torch.func.jvp` to build the detached flow-matching targets and records the loss terms needed for
analysis.

To visualize VFMF trajectories, use `flowviz.pipelines.flow_matching.compute_variational_forward_mean_trajectories`. The helper samples latent
codes from the standard normal prior (matching inference-time behaviour) and integrates the velocity field with the Euler integrator,
producing a tensor of states and their corresponding time stamps that can be plotted with the existing visualization utilities.

## Variational Forward Mean Flow Modified (VFMF-M)

The modified VFMF objective leverages the same latent-conditioned velocity network but replaces the encoder with a forward-only variant that
conditions exclusively on \(x_0, x_1\). Consequently, the latent time derivative is identically zero and no Jacobian-vector products are needed for
the encoder pathway. Training is handled by `flowviz.pipelines.flow_matching.train_variational_forward_mean_flow_modified_matching`, which mirrors
the bookkeeping used for VFMF while omitting the latent JVP computation.

For visualization, reuse `flowviz.pipelines.flow_matching.compute_variational_forward_mean_trajectories`—the inference procedure samples latent codes
from the prior and integrates the velocity network exactly as in the standard VFMF case, ensuring comparable trajectory outputs.

## Adding New Flow Variants

To extend the project with a new flow-matching variant:

1. Implement a new objective by subclassing `flowviz.flows.base.FlowMatchingObjective`.
2. Either reuse the provided training utilities (`flowviz.training.trainer.train_model` for deterministic objectives or
   `flowviz.pipelines.flow_matching.train_variational_flow_matching` as an example for latent-variable training), or create
   a custom training loop following the same interface.
3. Reuse the plotting utilities or create new ones under `flowviz.visualization` as needed.

This modular approach keeps the training, simulation, and visualization layers reusable across future experiments.
