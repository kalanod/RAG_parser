"""Logging configuration shared across the project."""
from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_LEVEL = logging.INFO


def configure(level: int = _DEFAULT_LEVEL) -> None:
    """Configure the root logger."""

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger, configuring logging if required."""

    configure()
    return logging.getLogger(name)

