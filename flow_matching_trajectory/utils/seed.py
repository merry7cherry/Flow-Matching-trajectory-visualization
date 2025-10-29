"""Utility helpers for reproducibility."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SeedConfig:
    """Container holding random seed configuration."""

    seed: int = 42
    deterministic: bool = True
    enable_tf32: bool = False


def set_global_seed(config: Optional[SeedConfig] = None) -> int:
    """Configure deterministic behaviour across Python, NumPy and PyTorch.

    Args:
        config: Optional seed configuration. If ``None`` a default configuration
            with seed ``42`` is used.

    Returns:
        The seed value that has been applied.
    """

    if config is None:
        config = SeedConfig()

    seed = config.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if not config.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return seed
