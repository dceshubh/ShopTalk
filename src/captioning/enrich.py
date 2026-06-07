"""Top-level captioning pipeline: sample -> fetch images -> caption -> rebuild canonical
docs with a `visual:` segment -> persist `products_enriched.parquet`. Mirrors
`src.preprocess.build` as the single, reproducible entry point for "enrich the catalog
with visual captions" — the EDA notebook, comparison harness, and any re-run all call this.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.captioning.caption import Captioner, load_captioner
from src.captioning.images import ensure_images_cached
from src.common.config import load_config, resolve_path
from src.common.logging import get_logger
from src.common.timer import Timer
from src.preprocess.clean import build_doc_text

logger = get_logger(__name__)


def caption_products(
    df: pd.DataFrame,
    captioner: Captioner,
    cache_dir: Path,
    *,
    max_new_tokens: int = 30,
) -> pd.DataFrame:
    """Caption every product in `df`, returning a copy with a `visual_caption` column and
    `doc_text` rebuilt to include the `visual: ...` segment — via the *same* `build_doc_text`
    the offline indexer and online API both call (see `src/preprocess/clean.py`), so a
    captioned product's document shape is identical regardless of when the caption arrived.

    A product whose image fails to download or caption keeps `visual_caption=None` and its
    original (uncaptioned) `doc_text` — logged as a failure, not silently dropped, so the
    "100% non-empty caption, retries/fallbacks logged" exit-gate can be checked against an
    explicit failure count rather than an assumption.
    """
    image_paths = list(dict.fromkeys(df["image_path"]))  # de-duped, order-preserving
    local_paths = ensure_images_cached(image_paths, cache_dir)

    captions: list[str | None] = []
    failures = 0
    total = len(df)
    with Timer(f"caption_products[{captioner.model_name}]"):
        for i, row in enumerate(df.itertuples(index=False), start=1):
            try:
                image = Image.open(local_paths[row.image_path]).convert("RGB")
                caption = captioner.caption(image, max_new_tokens=max_new_tokens)
            except Exception:
                logger.exception(
                    "Captioning failed for item_id=%s image_path=%s", row.item_id, row.image_path
                )
                caption = None
                failures += 1
            captions.append(caption)
            if i % 25 == 0 or i == total:
                logger.info("Captioned %d / %d (%d failures so far)", i, total, failures)

    out = df.copy()
    out["visual_caption"] = captions
    out["doc_text"] = [
        build_doc_text(
            name=r.name,
            brand=r.brand,
            product_type=r.product_type,
            color=r.color,
            material=r.material,
            bullet_points=list(r.bullet_points),
            keywords=list(r.keywords),
            visual_caption=r.visual_caption,
        )
        for r in out.itertuples(index=False)
    ]
    logger.info(
        "Captioned %d products with %s -- %d failures (%.1f%% non-empty)",
        len(out),
        captioner.model_name,
        failures,
        100 * (len(out) - failures) / len(out) if len(out) else 0.0,
    )
    return out


def build_enriched_dataset(
    products_path: Path | str | None = None,
    output_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    sample_size: int | None = None,
    model_name: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Caption a (sub)sample of the persisted catalog with the primary captioning model
    and persist the enriched result. Defaults are pulled from `configs/config.yaml`.

    `sample_size` defaults to `captioning.dev_sample_size` (~200) — sized for local MPS
    iteration per the local-first compute strategy in docs/ShopTalk_Plan.md §8. The full
    ~40K-product run is expected to execute on a free Kaggle/Colab GPU; pass a larger
    `sample_size` (or `len(products_df)`) there and pull the resulting parquet back here.
    """
    cfg = load_config()
    products_path = Path(products_path) if products_path else resolve_path(cfg["paths"]["products_parquet"])
    output_path = (
        Path(output_path) if output_path else resolve_path(cfg["paths"]["products_enriched_parquet"])
    )
    cache_dir = Path(cache_dir) if cache_dir else resolve_path(cfg["paths"]["data_dir"]) / "images"
    cap_cfg = cfg["captioning"]
    sample_size = sample_size if sample_size is not None else cap_cfg["dev_sample_size"]
    model_name = model_name or cfg["models"]["captioning"]["primary"]
    seed = seed if seed is not None else cfg["dataset"]["random_seed"]

    df = pd.read_parquet(products_path)
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    captioner = load_captioner(model_name)
    enriched = caption_products(df, captioner, cache_dir, max_new_tokens=cap_cfg["max_new_tokens"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    logger.info("Wrote %d enriched products to %s", len(enriched), output_path)
    return enriched


if __name__ == "__main__":
    build_enriched_dataset()
