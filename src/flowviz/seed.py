import os
import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_generator(seed: int, device: Optional[str] = None) -> torch.Generator:
    generator = torch.Generator(device=device or "cpu")
    generator.manual_seed(seed)
    return generator
