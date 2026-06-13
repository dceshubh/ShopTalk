"""Synthetic (query -> positive product) pair generation for Phase 4 LoRA fine-tuning
(docs/ShopTalk_Plan.md §2.3).

Templates each product's OWN structured attributes (`product_type`, `color`, `material`,
`brand`) into shopper-style queries — e.g. "wool blue home furniture and decor" — so
training pairs need no human labels. These templates are deliberately a different STYLE
from the hand-written `data/eval/golden_set.json` queries: `triplet_mining.assert_eval_integrity`
checks the two never overlap, so "fine-tuned beats pretrained" can't be an artifact of the
fine-tune merely learning this template.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import pandas as pd

# ABO attribute values are sometimes suffixed with a numeric color code, e.g. "Black_00826".
_TRAILING_CODE_RE = re.compile(r"_\d+$")


@dataclass(frozen=True)
class QueryProductPair:
    """One synthetic (query -> positive product) training pair."""

    query: str
    positive_item_id: str


def _humanize(value: str) -> str:
    """`"HOME_FURNITURE_AND_DECOR"` -> `"home furniture and decor"`; `"Black_00826"` ->
    `"black"` — strips ABO's numeric color-code suffixes and normalizes casing/underscores
    so templated queries read like natural shopper phrases."""
    value = _TRAILING_CODE_RE.sub("", value)
    return value.replace("_", " ").strip().lower()


def templates_for_product(row: pd.Series) -> list[str]:
    """Every distinct shopper-style query this product's own populated attributes support.

    `product_type` is always present and is itself a valid (if generic) query; `color`,
    `material`, and `brand` each add more specific variants when populated. Order matters
    only for de-duplication (`generate_synthetic_pairs` samples from this list).
    """
    product_type = _humanize(row["product_type"])
    color = _humanize(row["color"]) if row.get("color") else None
    material = _humanize(row["material"]) if row.get("material") else None
    brand = row.get("brand")

    queries = [product_type]
    if color:
        queries.append(f"{color} {product_type}")
    if material:
        queries.append(f"{material} {product_type}")
    if color and material:
        queries.append(f"{material} {color} {product_type}")
    if brand:
        queries.append(f"{brand.lower()} {product_type}")

    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            unique.append(query)
    return unique


def generate_synthetic_pairs(
    df: pd.DataFrame, *, queries_per_product: int = 2, seed: int = 42
) -> list[QueryProductPair]:
    """Sample up to `queries_per_product` template queries per product.

    Sampling (rather than using every template) keeps the training set's attribute-
    combination distribution from being dominated by products with many populated
    attributes — every product contributes a comparable number of pairs.
    """
    rng = random.Random(seed)
    pairs: list[QueryProductPair] = []
    for _, row in df.iterrows():
        candidates = templates_for_product(row)
        chosen = rng.sample(candidates, k=min(queries_per_product, len(candidates)))
        pairs.extend(QueryProductPair(query=query, positive_item_id=row["item_id"]) for query in chosen)
    return pairs
