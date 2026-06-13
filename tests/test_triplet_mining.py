"""Unit tests for src.embeddings.finetune.triplet_mining — assembling
`(query, positive_item_id, negative_item_id)` training triplets for Phase 4 LoRA
fine-tuning, and the eval-integrity gate that keeps "fine-tuned beats pretrained" from
being an artifact of training on the golden set (docs/ShopTalk_Plan.md §2.3, Phase 4 exit
gates).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.embeddings.finetune.query_generation import QueryProductPair
from src.embeddings.finetune.triplet_mining import (
    assert_eval_integrity,
    build_training_triplets,
    exclude_golden_set_products,
    mine_attribute_hard_negatives,
)
from src.eval.hard_negatives import HardNegativeTriplet
from src.eval.harness import GoldenCase


def _products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "WALNUT_TABLE",
                "name": "Walnut Coffee Table",
                "brand": "Acme",
                "product_type": "TABLE",
                "color": "Walnut",
                "material": "Wood",
                "bullet_points": [],
                "keywords": [],
            },
            {
                "item_id": "TEAL_TABLE",
                "name": "Teal Side Table",
                "brand": "Acme",
                "product_type": "TABLE",
                "color": "Teal",
                "material": "Wood",
                "bullet_points": [],
                "keywords": [],
            },
            {
                "item_id": "ONLY_CHAIR",
                "name": "Lonely Chair",
                "brand": None,
                "product_type": "CHAIR",
                "color": "Black",
                "material": None,
                "bullet_points": [],
                "keywords": [],
            },
        ]
    )


def _golden_cases() -> list[GoldenCase]:
    return [
        GoldenCase(
            query="a walnut coffee table", relevant_item_ids=frozenset({"WALNUT_TABLE"}), category="easy"
        ),
    ]


# ---------------------------------------------------------------------------
# exclude_golden_set_products
# ---------------------------------------------------------------------------


def test_exclude_golden_set_products_drops_golden_relevant_ids():
    df = _products_df()

    trainable = exclude_golden_set_products(df, _golden_cases())

    assert "WALNUT_TABLE" not in trainable["item_id"].tolist()
    assert {"TEAL_TABLE", "ONLY_CHAIR"} == set(trainable["item_id"].tolist())


# ---------------------------------------------------------------------------
# mine_attribute_hard_negatives
# ---------------------------------------------------------------------------


def test_mines_a_same_type_different_attribute_product_as_hard_negative():
    df = _products_df()
    pairs = [QueryProductPair(query="walnut table", positive_item_id="WALNUT_TABLE")]

    [triplet] = mine_attribute_hard_negatives(df, pairs)

    assert triplet.query == "walnut table"
    assert triplet.positive_item_id == "WALNUT_TABLE"
    assert triplet.negative_item_id == "TEAL_TABLE"  # same product_type, different color


def test_a_positive_with_no_qualifying_sibling_yields_no_triplet():
    df = _products_df()
    pairs = [QueryProductPair(query="black chair", positive_item_id="ONLY_CHAIR")]

    assert mine_attribute_hard_negatives(df, pairs) == []


def test_a_pair_referencing_an_unknown_item_id_is_skipped():
    df = _products_df()
    pairs = [QueryProductPair(query="mystery item", positive_item_id="NOT_IN_DF")]

    assert mine_attribute_hard_negatives(df, pairs) == []


# ---------------------------------------------------------------------------
# assert_eval_integrity
# ---------------------------------------------------------------------------


def test_assert_eval_integrity_passes_for_clean_triplets():
    triplets = [
        HardNegativeTriplet(query="teal table", positive_item_id="TEAL_TABLE", negative_item_id="ONLY_CHAIR")
    ]

    assert_eval_integrity(triplets, _golden_cases())  # does not raise


def test_assert_eval_integrity_rejects_a_golden_product_as_positive():
    triplets = [
        HardNegativeTriplet(
            query="teal table", positive_item_id="WALNUT_TABLE", negative_item_id="ONLY_CHAIR"
        )
    ]

    with pytest.raises(AssertionError, match="leaked into training triplets"):
        assert_eval_integrity(triplets, _golden_cases())


def test_assert_eval_integrity_rejects_a_query_matching_a_golden_query_verbatim():
    triplets = [
        HardNegativeTriplet(
            query="a walnut coffee table", positive_item_id="TEAL_TABLE", negative_item_id="ONLY_CHAIR"
        )
    ]

    with pytest.raises(AssertionError, match="duplicates a golden-set query"):
        assert_eval_integrity(triplets, _golden_cases())


# ---------------------------------------------------------------------------
# build_training_triplets (end-to-end)
# ---------------------------------------------------------------------------


def test_build_training_triplets_excludes_golden_products_end_to_end():
    df = _products_df()

    triplets = build_training_triplets(df, _golden_cases(), queries_per_product=2)

    golden_ids = {"WALNUT_TABLE"}
    for triplet in triplets:
        assert triplet.positive_item_id not in golden_ids
        assert triplet.negative_item_id not in golden_ids


def test_build_training_triplets_folds_in_feedback_triplets():
    df = _products_df()
    feedback_triplets = [
        HardNegativeTriplet(
            query="teal table for office", positive_item_id="TEAL_TABLE", negative_item_id="ONLY_CHAIR"
        )
    ]

    triplets = build_training_triplets(
        df, _golden_cases(), feedback_triplets=feedback_triplets, queries_per_product=0
    )

    assert triplets == feedback_triplets


def test_build_training_triplets_drops_feedback_triplets_referencing_golden_products():
    df = _products_df()
    feedback_triplets = [
        HardNegativeTriplet(
            query="some query", positive_item_id="WALNUT_TABLE", negative_item_id="ONLY_CHAIR"
        )
    ]

    triplets = build_training_triplets(
        df, _golden_cases(), feedback_triplets=feedback_triplets, queries_per_product=0
    )

    assert triplets == []
