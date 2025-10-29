from dataclasses import dataclass


@dataclass
class VariationalFlowConfig:
    latent_dim: int = 8
    kl_weight: float = 1.0
    matching_weight: float = 1.0
    reconstruction_weight: float = 1.0
    encoder_hidden_sizes: tuple[int, ...] = (128, 128)
    velocity_hidden_sizes: tuple[int, ...] = (128, 128, 128)


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
