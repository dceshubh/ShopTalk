"""Unit tests for src.embeddings.finetune.query_generation — templating synthetic
(query -> positive product) training pairs from a product's own attributes
(docs/ShopTalk_Plan.md §2.3).
"""

from __future__ import annotations

import pandas as pd

from src.embeddings.finetune.query_generation import (
    QueryProductPair,
    generate_synthetic_pairs,
    templates_for_product,
)


def _row(**overrides) -> pd.Series:
    base = {
        "item_id": "B0TEST0001",
        "product_type": "CHAIR",
        "color": None,
        "material": None,
        "brand": None,
    }
    base.update(overrides)
    return pd.Series(base)


def test_product_type_alone_is_always_a_template():
    assert templates_for_product(_row()) == ["chair"]


def test_color_and_material_each_add_a_specific_variant():
    templates = templates_for_product(_row(color="Blue", material="Wool"))

    assert "chair" in templates
    assert "blue chair" in templates
    assert "wool chair" in templates
    assert "wool blue chair" in templates


def test_brand_adds_a_brand_plus_type_variant():
    templates = templates_for_product(_row(brand="Stone & Beam"))

    assert "stone & beam chair" in templates


def test_trailing_numeric_color_codes_are_stripped():
    templates = templates_for_product(_row(color="Black_00826"))

    assert "black chair" in templates
    assert "black_00826 chair" not in templates


def test_templates_are_deduplicated():
    # color == material, after humanizing, would otherwise produce duplicate variants
    templates = templates_for_product(_row(color="Wood", material="Wood"))

    assert len(templates) == len(set(templates))


def test_generate_synthetic_pairs_samples_up_to_queries_per_product():
    df = pd.DataFrame(
        [
            _row(item_id="B0TEST0001", color="Blue", material="Wool"),
            _row(item_id="B0TEST0002"),  # only one template available
        ]
    )

    pairs = generate_synthetic_pairs(df, queries_per_product=2, seed=42)

    by_product: dict[str, list[QueryProductPair]] = {}
    for pair in pairs:
        by_product.setdefault(pair.positive_item_id, []).append(pair)

    assert len(by_product["B0TEST0001"]) == 2
    assert len(by_product["B0TEST0002"]) == 1  # capped by the number of available templates


def test_generate_synthetic_pairs_is_deterministic_for_a_fixed_seed():
    df = pd.DataFrame([_row(color="Blue", material="Wool")])

    first = generate_synthetic_pairs(df, queries_per_product=2, seed=42)
    second = generate_synthetic_pairs(df, queries_per_product=2, seed=42)

    assert first == second


def test_every_pair_references_a_real_item_id_and_non_empty_query():
    df = pd.DataFrame([_row(color="Blue", material="Wool")])

    for pair in generate_synthetic_pairs(df, queries_per_product=2):
        assert pair.positive_item_id == "B0TEST0001"
        assert isinstance(pair.query, str) and pair.query
