"""Capped-stratified subsampling.

EDA on the full English-filtered catalog (~123K products) showed `CELLULAR_PHONE_CASE` alone
accounts for roughly half of it. A plain proportional-stratified sample would reproduce that
skew — a ~40K sample would still be ~50% phone cases, leaving the demo, golden evaluation set,
and fine-tuning data thin on every other category.

We therefore cap any single `product_type`'s share of the sample (`max_category_fraction`)
and redistribute the freed budget proportionally across the remaining categories. This is a
documented trade-off: the sample is no longer perfectly representative of the raw catalog
distribution (which we report separately, as-is, in the EDA), but it is far more useful for
building and evaluating a retrieval system that needs to demonstrate breadth across product
categories and attributes.
"""

from __future__ import annotations

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CATEGORY_FRACTION = 0.10


def capped_stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    *,
    category_col: str = "product_type",
    max_category_fraction: float = DEFAULT_MAX_CATEGORY_FRACTION,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample up to `target_size` rows from `df`, capping each category's share.

    Algorithm (documented for reproducibility):
      1. `cap = max_category_fraction * target_size` — the maximum rows any one category
         may contribute.
      2. Each category is allocated `min(raw_count, cap)`. This respects the cap AND never
         asks a category for more rows than it has — both constraints fall out of one
         expression, so there is no separate "small vs. large category" branch to keep
         in sync.
      3. If the capped allocations already sum to more than `target_size` (i.e. there are
         enough categories to fill the budget without any of them dominating), every
         allocation is scaled down proportionally to land at `target_size`. The relative
         shape — which is what the cap is protecting — is preserved by a uniform scale.
      4. If the capped allocations sum to less than `target_size`, that's the final answer:
         every category is already giving everything it can without breaching the cap or
         running out of rows, so the sample is smaller than `target_size` (documented as
         "approximately" in the function's contract — favoring category balance over
         hitting the raw number exactly).

    Returns a new DataFrame (rows sampled with `random_state=seed`, so the result is
    reproducible — required by the "deterministic preprocessing" exit-gate check).
    """
    if target_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = df[category_col].value_counts()
    cap = int(max_category_fraction * target_size)

    allocation = {category: min(int(count), cap) for category, count in counts.items()}
    total_allocated = sum(allocation.values())

    if total_allocated > target_size:
        scale = target_size / total_allocated
        allocation = {category: int(n * scale) for category, n in allocation.items()}

    parts: list[pd.DataFrame] = []
    for category, n in allocation.items():
        if n > 0:
            parts.append(df[df[category_col] == category].sample(n=n, random_state=seed))

    if not parts:
        return df.iloc[0:0].reset_index(drop=True)

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Capped-stratified sample: %d rows from %d (target=%d, cap=%.0f%% -> %d rows/category)",
        len(sampled),
        len(df),
        target_size,
        max_category_fraction * 100,
        cap,
    )
    return sampled
