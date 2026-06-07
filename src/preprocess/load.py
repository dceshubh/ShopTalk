"""Loads raw ABO listing shards (gzip-compressed JSON Lines) and the image join table.

Kept separate from cleaning/canonicalization so the "what the raw data looks like" concern
stays isolated from the "how we turn it into a canonical doc" concern.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)

LISTINGS_GLOB = "listings/metadata/listings_*.json.gz"
IMAGES_CSV = "images/metadata/images.csv"


def iter_raw_listings(raw_dir: Path | str) -> Iterator[dict[str, Any]]:
    """Yield raw listing records (one dict per product) across all shards, in shard order."""
    raw_dir = Path(raw_dir)
    shard_paths = sorted(raw_dir.glob(LISTINGS_GLOB))
    if not shard_paths:
        raise FileNotFoundError(f"No listing shards found under {raw_dir / LISTINGS_GLOB}")

    for shard_path in shard_paths:
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_image_index(raw_dir: Path | str) -> dict[str, str]:
    """Return {image_id: relative_path} built from images.csv (~398K entries, ~6 MB).

    We deliberately consume only this small join table — never the multi-GB image archive —
    so EDA and preprocessing stay fast and disk-light; captioning (which needs the actual
    image bytes) runs separately, on a GPU, against a subsampled product list.
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / IMAGES_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"images.csv not found at {csv_path}")

    index: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        next(f)  # header: image_id,height,width,path
        for line in f:
            image_id, _height, _width, path = line.rstrip("\n").split(",", 3)
            index[image_id] = path

    logger.info("Loaded image index: %d entries", len(index))
    return index
