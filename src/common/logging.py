"""Process-wide logging setup, configured from config.yaml so format/level stay consistent
across the offline pipeline and the online API (same module → "documented, reproducible setup")."""

from __future__ import annotations

import logging

from src.common.config import load_config

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    cfg = load_config()["logging"]
    logging.basicConfig(level=cfg["level"], format=cfg["format"])
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
