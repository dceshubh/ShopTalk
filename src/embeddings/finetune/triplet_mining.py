"""Assembles `(query, positive_item_id, negative_item_id)` training triplets for Phase 4
LoRA fine-tuning (docs/ShopTalk_Plan.md §2.3), from two sources:

  1. **Attribute hard negatives**: for each synthetic (query -> positive product) pair from
     `query_generation.generate_synthetic_pairs`, find another product of the SAME
     `product_type` but a DIFFERENT `color`/`material` — "a teal table is a hard negative
     for 'walnut table'" (§2.3). This is the signal that teaches fine-grained attribute
     discrimination, the exact thing pretrained models miss.
  2. **Real feedback hard negatives** (`src.eval.hard_negatives.mine_hard_negatives`) —
     already in the same `(query, positive_item_id, negative_item_id)` shape, closing the
     Phase 9 -> Phase 4 loop.

Golden-set products are excluded from (1) before mining, and `assert_eval_integrity` checks
the combined result against the §2.3 "eval-integrity condition": golden-set products never
appear in a training triplet, and no training query duplicates a hand-written golden query
verbatim.
"""

from __future__ import annotations

import random

import pandas as pd

from src.embeddings.finetune.query_generation import QueryProductPair, generate_synthetic_pairs
from src.eval.hard_negatives import HardNegativeTriplet
from src.eval.harness import GoldenCase


def _golden_item_ids(golden_cases: list[GoldenCase]) -> set[str]:
    return {item_id for case in golden_cases for item_id in case.relevant_item_ids}


def exclude_golden_set_products(df: pd.DataFrame, golden_cases: list[GoldenCase]) -> pd.DataFrame:
    """Drop every product that appears as a `relevant_item_id` in the golden set — the
    Phase-4 exit gate's "golden-set products also excluded from mining"."""
    golden_ids = _golden_item_ids(golden_cases)
    return df[~df["item_id"].isin(golden_ids)].reset_index(drop=True)


def mine_attribute_hard_negatives(
    df: pd.DataFrame, pairs: list[QueryProductPair], *, seed: int = 42
) -> list[HardNegativeTriplet]:
    """For each synthetic (query, positive) pair, pick a same-`product_type`, different-
    `color`-or-`material` product as the hard negative.

    Pairs whose positive has no qualifying same-type sibling are skipped (not force-paired
    with an unrelated item — see `src.eval.hard_negatives` for the same "skip, don't force"
    rationale applied to feedback-derived triplets).
    """
    rng = random.Random(seed)
    by_id = df.set_index("item_id")
    by_type: dict[str, list[str]] = {}
    for item_id, row in by_id.iterrows():
        by_type.setdefault(row["product_type"], []).append(item_id)

    triplets: list[HardNegativeTriplet] = []
    for pair in pairs:
        if pair.positive_item_id not in by_id.index:
            continue
        positive = by_id.loc[pair.positive_item_id]
        candidates = [
            other_id
            for other_id in by_type.get(positive["product_type"], [])
            if other_id != pair.positive_item_id
            and (
                by_id.loc[other_id]["color"] != positive["color"]
                or by_id.loc[other_id]["material"] != positive["material"]
            )
        ]
        if not candidates:
            continue
        negative_id = rng.choice(candidates)
        triplets.append(
            HardNegativeTriplet(
                query=pair.query,
                positive_item_id=pair.positive_item_id,
                negative_item_id=negative_id,
            )
        )
    return triplets


def assert_eval_integrity(triplets: list[HardNegativeTriplet], golden_cases: list[GoldenCase]) -> None:
    """Phase-4 exit gate, the literal assertion: no training triplet references a
    golden-set product (as positive OR negative), and no training query duplicates a
    hand-written golden query verbatim. Raises `AssertionError` on the first violation —
    this is a correctness gate, not a soft metric.
    """
    golden_ids = _golden_item_ids(golden_cases)
    golden_queries = {case.query.strip().lower() for case in golden_cases}

    for triplet in triplets:
        assert (
            triplet.positive_item_id not in golden_ids
        ), f"Golden-set product {triplet.positive_item_id!r} leaked into training triplets as a positive"
        assert (
            triplet.negative_item_id not in golden_ids
        ), f"Golden-set product {triplet.negative_item_id!r} leaked into training triplets as a negative"
        assert (
            triplet.query.strip().lower() not in golden_queries
        ), f"Training query {triplet.query!r} duplicates a golden-set query verbatim"


def build_training_triplets(
    df: pd.DataFrame,
    golden_cases: list[GoldenCase],
    *,
    feedback_triplets: list[HardNegativeTriplet] | None = None,
    queries_per_product: int = 2,
    seed: int = 42,
) -> list[HardNegativeTriplet]:
    """End-to-end: exclude golden-set products, generate synthetic (query, positive) pairs,
    mine attribute hard negatives, fold in real feedback hard negatives (any referencing a
    golden-set product are dropped), and assert eval integrity on the result.
    """
    trainable_df = exclude_golden_set_products(df, golden_cases)
    pairs = generate_synthetic_pairs(trainable_df, queries_per_product=queries_per_product, seed=seed)
    triplets = mine_attribute_hard_negatives(trainable_df, pairs, seed=seed)

    golden_ids = _golden_item_ids(golden_cases)
    for triplet in feedback_triplets or []:
        if triplet.positive_item_id not in golden_ids and triplet.negative_item_id not in golden_ids:
            triplets.append(triplet)

    assert_eval_integrity(triplets, golden_cases)
    return triplets
