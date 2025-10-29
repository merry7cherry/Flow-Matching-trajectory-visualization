# Flow Matching Trajectory Visualization

This project visualizes flow matching trajectories for 1D and 2D synthetic datasets. It provides a modular codebase that makes it straightforward to add new flow-matching variants, models, and datasets.

## Features

- **Standard Flow Matching** with linear interpolation.
- **Rectified Flow** training using the trajectories produced by the base flow-matching model.
- **Variational Flow Matching (VFM)** with a latent-conditioned velocity field trained jointly with a variational encoder.
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

   Images for the 1D and 2D experiments will be saved under the `outputs/` directory. Use the following optional flags to tune
   the variational flow experiment:

   - `--variational-latent-dim`: dimensionality of the latent code sampled from the VAE prior (default: 8)
   - `--variational-kl-weight`: weight for the KL divergence between encoder posterior and standard normal prior (default: 1.0)
   - `--variational-matching-weight`: scales the \(\|v\|^2-\Delta^X\) regularizer (default: 1.0)
   - `--variational-reconstruction-weight`: scales the reconstruction-style MSE penalty (default: 1.0)

## Adding New Flow Variants

To extend the project with a new flow-matching variant:

1. Implement a new objective by subclassing `flowviz.flows.base.FlowMatchingObjective`.
2. Either reuse the provided training utilities (`flowviz.training.trainer.train_model` for deterministic objectives or
   `flowviz.pipelines.flow_matching.train_variational_flow_matching` as an example for latent-variable training), or create
   a custom training loop following the same interface.
3. Reuse the plotting utilities or create new ones under `flowviz.visualization` as needed.

This modular approach keeps the training, simulation, and visualization layers reusable across future experiments.
