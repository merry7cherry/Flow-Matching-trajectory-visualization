from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Type

from .data.base import PairDataset
from .data.synthetic import GaussianMixture1D, GaussianMixture2D


@dataclass
class VariationalFlowConfig:
    latent_dim: int = 8
    kl_weight: float = 1.0
    matching_weight: float = 1.0
    encoder_hidden_sizes: tuple[int, ...] = (128, 128)
    velocity_hidden_sizes: tuple[int, ...] = (128, 128, 128)


@dataclass
class VariationalMeanFlowConfig(VariationalFlowConfig):
    """Configuration for training and sampling variational mean flows."""


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 256
    steps_per_epoch: int = 100
    learning_rate: float = 1e-3
    device: str = "cpu"


@dataclass
class IntegratorConfig:
    num_steps: int = 50


@dataclass
class RectifiedFlowConfig:
    num_samples: int = 4096
    batch_size: int = 512


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for constructing synthetic pair datasets."""

    name: str
    label: str
    dataset_cls: Type[PairDataset]
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def create_dataset(self, seed: int) -> PairDataset:
        params: Dict[str, Any] = dict(self.kwargs)
        params.setdefault("seed", seed)
        return self.dataset_cls(**params)


DEFAULT_1D_DATASET = DatasetConfig(
    name="1d_default",
    label="1D Gaussian Mixture",
    dataset_cls=GaussianMixture1D,
    kwargs={
        "means": (-2.0, 2.0),
        "std": 0.35,
        "base_std": 0.5,
    },
)

DEFAULT_2D_DATASET = DatasetConfig(
    name="2d_default",
    label="2D Gaussian Mixture",
    dataset_cls=GaussianMixture2D,
    kwargs={
        "centers": ((-4.0, -4.0), (-4.0, 4.0), (4.0, -4.0), (4.5, 4.5)),
        "std": 0.4,
        "base_std": 0.6,
    },
)

WIDE_SOURCE_NARROW_TARGET_1D = DatasetConfig(
    name="1d_wide_source_narrow_target",
    label="1D Wide Source / Narrow Target",
    dataset_cls=GaussianMixture1D,
    kwargs={
        "means": (-4.5, 4.5),
        "std": 0.18,
        "base_std": 1.5,
    },
)

HEXAGONAL_TARGET_2D = DatasetConfig(
    name="2d_hexagonal_target",
    label="2D Hexagonal Target",
    dataset_cls=GaussianMixture2D,
    kwargs={
        "centers": (
            (4.0, 0.0),
            (2.0, 3.464),
            (-2.0, 3.464),
            (-4.0, 0.0),
            (-2.0, -3.464),
            (2.0, -3.464),
        ),
        "std": 0.3,
        "base_std": 0.6,
    },
)

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    cfg.name: cfg
    for cfg in (
        DEFAULT_1D_DATASET,
        DEFAULT_2D_DATASET,
        WIDE_SOURCE_NARROW_TARGET_1D,
        HEXAGONAL_TARGET_2D,
    )
}


__all__ = [
    "VariationalFlowConfig",
    "VariationalMeanFlowConfig",
    "TrainingConfig",
    "IntegratorConfig",
    "RectifiedFlowConfig",
    "DatasetConfig",
    "DEFAULT_1D_DATASET",
    "DEFAULT_2D_DATASET",
    "WIDE_SOURCE_NARROW_TARGET_1D",
    "HEXAGONAL_TARGET_2D",
    "DATASET_CONFIGS",
]
