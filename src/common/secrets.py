"""Loads `.env` (gitignored, see `.env.example`) and exposes secrets/connection strings —
the one place that touches `os.environ` so no module reaches for it directly.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.common.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def get_env(key: str, *, default: str | None = None, required: bool = False) -> str | None:
    """Read an environment variable (after loading `.env`).

    Raises with a pointer to `.env.example` when a `required` key is missing — the failure
    a misconfigured local/CI run should surface immediately, not three calls deep as a
    cryptic 401 from the provider.
    """
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable {key!r} — copy .env.example to .env and fill it in."
        )
    return value
