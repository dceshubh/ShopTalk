"""Fetches individual ABO product images directly from the public S3 bucket.

ABO images are distributed as a 3 GB tar (`abo-images-small.tar`), but every image inside
it is *also* directly addressable as an individual S3 object at the same relative path —
confirmed with a HEAD request against `images/small/<path>` returning 200 OK. This lets us
pull just the handful of images a given run needs (a 200-image dev sample, a handful of
demo examples) without ever unpacking the full archive locally — see
docs/ShopTalk_Plan.md §8 ("do NOT unpack the 3 GB ABO image tar locally").
"""

from __future__ import annotations

from pathlib import Path

import httpx

from src.common.logging import get_logger

logger = get_logger(__name__)

ABO_IMAGES_BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small"


def fetch_image(image_path: str, cache_dir: Path, *, timeout: float = 30.0) -> Path:
    """Download `images/small/<image_path>` into `cache_dir`, returning the local path.

    Idempotent — if the file is already cached, no network request is made. `image_path`
    keeps its `<2-char-prefix>/<id>.jpg` shape inside `cache_dir`, so the cache mirrors the
    join table's layout and stays trivially inspectable (and re-runnable: re-fetching a
    populated cache is a no-op scan, not a re-download).
    """
    local_path = cache_dir / image_path
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ABO_IMAGES_BASE_URL}/{image_path}"
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    return local_path


def ensure_images_cached(image_paths: list[str], cache_dir: Path) -> dict[str, Path]:
    """Fetch every path in `image_paths` (de-duped by the caller; cached ones are a no-op),
    returning a `{image_path: local_path}` map. These are one-at-a-time HTTP GETs against
    S3 — not a bulk operation — so we log progress periodically rather than per-file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    total = len(image_paths)
    for i, image_path in enumerate(image_paths, start=1):
        resolved[image_path] = fetch_image(image_path, cache_dir)
        if i % 50 == 0 or i == total:
            logger.info("Image cache: %d / %d resolved (cache_dir=%s)", i, total, cache_dir)
    return resolved
