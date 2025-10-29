# Flow Matching Trajectory Visualisation

This repository provides a modular implementation of standard Flow Matching for
one- and two-dimensional synthetic datasets together with tools to visualise the
generation trajectories.

## Project layout

```
src/flow_matching_viz/
  data/             Synthetic datasets (1D Gaussian mixture and 2D two moons)
  flows/            Flow Matching abstractions and rollout utilities
  models/           Neural network architectures
  training/         Optimisation loops and objectives
  utils/            Helper samplers and seeding utilities
  visualization/    Trajectory sampling helpers and plotting functions
scripts/
  visualize_standard_flow_matching.py  Entry point for training and visualisation
```

All package code lives under `src/` to keep the Python path clean. When running
scripts directly, prepend `PYTHONPATH=src` so that the package can be resolved.

## Quick start

1. Create a Python environment with PyTorch and Matplotlib installed.
2. Train the 1D and 2D Flow Matching models and generate plots:

   ```bash
   PYTHONPATH=src python scripts/visualize_standard_flow_matching.py \
     --output-dir outputs \
     --steps-1d 3000 --steps-2d 4000 \
     --batch-size 2048 --lr 3e-4
   ```

The script stores the trajectory visualisations in the configured output
folder (defaults to `outputs/`).

## Extending the project

The package structure isolates datasets, models, flow definitions and training
routines into separate modules. Implementing new Flow Matching variants only
requires adding additional subclasses or functions in the respective module and
wiring them up inside a new script or training routine.
