"""Top-level pipeline: raw shards -> canonical English catalog -> capped-stratified sample
-> persisted parquet. This is the single entry point both the EDA notebook and any future
re-indexing run should call, so "rebuild the dataset" is always one reproducible function.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import load_config, resolve_path
from src.common.logging import get_logger
from src.common.timer import Timer
from src.preprocess.clean import build_canonical_product
from src.preprocess.load import iter_raw_listings, load_image_index
from src.preprocess.sample import capped_stratified_sample

logger = get_logger(__name__)

PREFERRED_DOMAIN = "amazon.com"


def _dedupe_cross_marketplace_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse cross-marketplace re-listings down to one canonical row per `item_id`.

    ABO's documented uniqueness key is `(item_id, domain_name)`, not `item_id` alone — EDA
    surfaced ~360 item_ids that are cross-listed on multiple marketplaces (e.g. the same
    sheet set as both `amazon.com.au` and `primenow.amazon.com`) with near-identical text.
    Surfacing every cross-listing as a separate catalog entry would show end users the
    "same" product more than once and seed the embedding index with near-duplicate vectors
    that distort retrieval evaluation (precision@K becomes ambiguous when two hits are the
    same physical product). We therefore keep exactly one listing per `item_id`: the
    `amazon.com` listing when present (the largest, most consistently English-tagged
    marketplace), otherwise the alphabetically-first `domain_name` — a deterministic,
    reproducible tie-break that doesn't privilege any other marketplace.
    """
    df = df.copy()
    df["_domain_rank"] = (df["domain_name"] != PREFERRED_DOMAIN).astype(int)
    df = df.sort_values(["item_id", "_domain_rank", "domain_name"], kind="stable")
    df = df.drop_duplicates(subset="item_id", keep="first")
    return df.drop(columns="_domain_rank").reset_index(drop=True)


def build_canonical_dataframe(raw_dir: Path | str) -> pd.DataFrame:
    """Stream every raw listing through `build_canonical_product`, then apply two more
    catalog-level filters on top of the per-record English/typed filter:

      1. Drop products with no resolvable `image_path` — EDA found ~0.4% of the English+typed
         catalog has no `main_image_id` at all (not a join failure; the field is simply absent
         in the source). Captioning needs complete image coverage, so these are dropped here
         rather than letting them surface as an incomplete sample downstream.
      2. Collapse cross-marketplace duplicates via `_dedupe_cross_marketplace_listings` —
         see that function's docstring for the reasoning.

    Deterministic by construction: shard files are read in sorted order, records within a
    shard preserve file order, `build_canonical_product` is a pure function, and the dedupe
    tie-break is a stable sort — so the same `raw_dir` always yields a byte-identical
    `doc_text` column (the exit-gate hash check) and a unique `item_id` per row.
    """
    image_index = load_image_index(raw_dir)

    rows: list[dict] = []
    total = 0
    with Timer("build_canonical_dataframe"):
        for record in iter_raw_listings(raw_dir):
            total += 1
            product = build_canonical_product(record, image_index)
            if product is not None:
                rows.append(product.model_dump())

    df = pd.DataFrame(rows)
    english_typed = len(df)

    df = df[df["image_path"].notna()].reset_index(drop=True)
    with_image = len(df)

    df = _dedupe_cross_marketplace_listings(df)

    logger.info(
        "Canonical catalog: %d / %d raw records kept as English+typed (%.1f%%); "
        "%d dropped for no resolvable image; %d cross-marketplace duplicates collapsed "
        "-> %d unique products",
        english_typed,
        total,
        100 * english_typed / total if total else 0.0,
        english_typed - with_image,
        with_image - len(df),
        len(df),
    )
    return df


def build_products_dataset(
    raw_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    target_size: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Build the canonical catalog, draw the capped-stratified subsample, persist to parquet.

    Parameters default to `configs/config.yaml` (`paths.raw_listings`'s parent,
    `paths.products_parquet`, `dataset.subsample_size`, `dataset.random_seed`) so this can be
    re-run identically from a single command, per the rubric's reproducibility requirement.
    """
    cfg = load_config()
    raw_dir = Path(raw_dir) if raw_dir else resolve_path(cfg["paths"]["raw_dir"])
    output_path = Path(output_path) if output_path else resolve_path(cfg["paths"]["products_parquet"])
    target_size = target_size if target_size is not None else cfg["dataset"]["subsample_size"]
    seed = seed if seed is not None else cfg["dataset"]["random_seed"]

    full_df = build_canonical_dataframe(raw_dir)
    sampled_df = capped_stratified_sample(full_df, target_size=target_size, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_parquet(output_path, index=False)
    logger.info("Wrote %d products to %s", len(sampled_df), output_path)
    return sampled_df


if __name__ == "__main__":
    build_products_dataset()
