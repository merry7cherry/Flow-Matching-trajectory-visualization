"""Simple logging helpers used throughout the project."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        level: Logging level.
        name: Optional logger name. If ``None`` the root logger is configured.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
