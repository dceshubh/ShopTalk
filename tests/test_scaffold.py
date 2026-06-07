"""Scaffold smoke tests — proves the project skeleton is importable and wired together."""

import time

from src.common.config import load_config, resolve_path
from src.common.logging import get_logger
from src.common.timer import Timer


def test_package_imports():
    import src  # noqa: F401


def test_config_loads_expected_keys():
    cfg = load_config()
    for key in ("paths", "dataset", "models", "retrieval", "finetune", "eval", "agent", "api", "logging"):
        assert key in cfg


def test_resolve_path_is_under_project_root():
    p = resolve_path("data/raw/abo-listings")
    assert p.is_absolute()
    assert p.parts[-3:] == ("data", "raw", "abo-listings")


def test_logger_returns_named_logger():
    logger = get_logger("shoptalk.test")
    assert logger.name == "shoptalk.test"


def test_timer_measures_elapsed_time():
    with Timer("smoke_sleep", log=False) as t:
        time.sleep(0.01)
    assert t.elapsed_ms >= 10.0
