"""Unit tests for the shared preprocessing / canonical-document module.

These cover exactly the edge cases called out in the cleaning-decision docstrings in
src/preprocess/clean.py: multi-language fields, missing color/material/bullets, HTML +
unicode punctuation, and keyword de-duplication. Also covers the capped-stratified sampler.
"""

import pandas as pd
import pytest

from src.preprocess.clean import (
    build_canonical_product,
    build_doc_text,
    clean_text,
    dedupe_list,
    pick_color,
    pick_localized,
    pick_material,
)
from src.preprocess.sample import capped_stratified_sample

# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_strips_html_and_unescapes_entities():
    assert clean_text("<b>Soft &amp; cozy</b> throw") == "Soft & cozy throw"


def test_clean_text_normalizes_unicode_punctuation_and_whitespace():
    assert clean_text("Kid’s “cozy”  chair —  blue") == 'Kid\'s "cozy" chair - blue'


def test_clean_text_returns_none_for_none_and_empty():
    assert clean_text(None) is None
    assert clean_text("   ") is None


# ---------------------------------------------------------------------------
# pick_localized — locale-priority selection
# ---------------------------------------------------------------------------


def test_pick_localized_prefers_en_us_over_en_in():
    entries = [
        {"language_tag": "en_IN", "value": "Loafer shoes for ladies mujer"},
        {"language_tag": "en_US", "value": "Women's Leather Loafers"},
    ]
    assert pick_localized(entries) == "Women's Leather Loafers"


def test_pick_localized_falls_back_to_any_english_locale():
    entries = [
        {"language_tag": "de_DE", "value": "Bürostuhl"},
        {"language_tag": "en_SG", "value": "Office Chair"},
    ]
    assert pick_localized(entries) == "Office Chair"


def test_pick_localized_returns_none_when_no_english_entry():
    entries = [{"language_tag": "de_DE", "value": "Bürostuhl"}]
    assert pick_localized(entries) is None


def test_pick_localized_handles_missing_or_empty_field():
    assert pick_localized(None) is None
    assert pick_localized([]) is None


# ---------------------------------------------------------------------------
# pick_color — display vs. standardized split
# ---------------------------------------------------------------------------


def test_pick_color_returns_display_and_standardized():
    entries = [{"language_tag": "en_US", "value": "Spinnsol Cocoa", "standardized_values": ["Brown"]}]
    assert pick_color(entries) == ("Spinnsol Cocoa", "Brown")


def test_pick_color_handles_missing_standardized_values():
    entries = [{"language_tag": "en_US", "value": "Taupe", "standardized_values": None}]
    assert pick_color(entries) == ("Taupe", None)


def test_pick_color_handles_missing_field():
    assert pick_color(None) == (None, None)


# ---------------------------------------------------------------------------
# pick_material — title-casing of compound strings
# ---------------------------------------------------------------------------


def test_pick_material_title_cases_compound_string():
    entries = [{"language_tag": "en_US", "value": "velvet, hardwood frame, metal legs"}]
    assert pick_material(entries) == "Velvet, Hardwood Frame, Metal Legs"


# ---------------------------------------------------------------------------
# dedupe_list — keyword cleanup
# ---------------------------------------------------------------------------


def test_dedupe_list_drops_empties_and_dedupes_case_insensitively():
    assert dedupe_list(["Loafers", "loafers", "", None, "Block Heel"]) == ["Loafers", "Block Heel"]


# ---------------------------------------------------------------------------
# build_doc_text — canonical string assembly
# ---------------------------------------------------------------------------


def test_build_doc_text_assembles_documented_format():
    text = build_doc_text(
        name="Mid-Century Walnut Coffee Table",
        brand="Rivet",
        product_type="TABLE",
        color="Walnut",
        material="Wood",
        bullet_points=["Solid wood top", "Tapered legs"],
        keywords=["coffee table", "mid-century"],
    )
    assert text == (
        "Mid-Century Walnut Coffee Table · Rivet · type: Table · color: Walnut · material: Wood "
        "· Solid wood top Tapered legs · keywords: coffee table, mid-century"
    )


def test_build_doc_text_appends_visual_caption_when_present():
    text = build_doc_text(
        name="Phone Case",
        brand=None,
        product_type="CELLULAR_PHONE_CASE",
        color=None,
        material=None,
        bullet_points=[],
        keywords=[],
        visual_caption="a clear case with a marble pattern",
    )
    assert text.endswith("visual: a clear case with a marble pattern")


def test_build_doc_text_places_visual_caption_before_bullet_points_and_keywords():
    # Encoders truncate from the end — placing `visual: <caption>` ahead of the bulky
    # bullet_points/keywords blocks keeps it in the surviving prefix on long listings,
    # instead of being the first thing chopped off (see clean.py: build_doc_text docstring).
    text = build_doc_text(
        name="Mid-Century Walnut Chair",
        brand="Rivet",
        product_type="CHAIR",
        color="Walnut",
        material="Wood",
        bullet_points=["Solid wood frame"],
        keywords=["chair"],
        visual_caption="a wooden chair with armrests",
    )
    visual_idx = text.index("visual: a wooden chair with armrests")
    assert visual_idx < text.index("Solid wood frame")
    assert visual_idx < text.index("keywords: chair")


def test_build_doc_text_omits_missing_optional_fields():
    text = build_doc_text(
        name="Generic Item",
        brand=None,
        product_type="HOME",
        color=None,
        material=None,
        bullet_points=[],
        keywords=[],
    )
    assert text == "Generic Item · type: Home"


# ---------------------------------------------------------------------------
# build_canonical_product — end-to-end record handling
# ---------------------------------------------------------------------------


def _english_record(**overrides):
    record = {
        "item_id": "B000TEST01",
        "domain_name": "amazon.com",
        "item_name": [{"language_tag": "en_US", "value": "Mid-Century Walnut Coffee Table"}],
        "brand": [{"language_tag": "en_US", "value": "Rivet"}],
        "product_type": [{"value": "TABLE"}],
        "color": [{"language_tag": "en_US", "value": "Walnut", "standardized_values": ["Brown"]}],
        "material": [{"language_tag": "en_US", "value": "wood"}],
        "bullet_point": [
            {"language_tag": "en_US", "value": "Solid wood top"},
            {"language_tag": "en_US", "value": "Tapered legs"},
        ],
        "item_keywords": [
            {"language_tag": "en_US", "value": "coffee table"},
            {"language_tag": "en_US", "value": "Coffee Table"},
        ],
        "main_image_id": "IMG123",
    }
    record.update(overrides)
    return record


def test_build_canonical_product_happy_path():
    product = build_canonical_product(_english_record(), image_index={"IMG123": "ab/abc123.jpg"})
    assert product is not None
    assert product.item_id == "B000TEST01"
    assert product.name == "Mid-Century Walnut Coffee Table"
    assert product.color == "Brown"  # standardized value preferred for the structured field
    assert product.material == "Wood"
    assert product.keywords == ["coffee table"]  # case-insensitive de-dupe
    assert product.image_path == "ab/abc123.jpg"
    assert "Mid-Century Walnut Coffee Table" in product.doc_text
    assert "visual:" not in product.doc_text  # no caption at preprocessing time


def test_build_canonical_product_drops_non_english_record():
    record = _english_record(item_name=[{"language_tag": "de_DE", "value": "Couchtisch"}])
    assert build_canonical_product(record) is None


def test_build_canonical_product_drops_record_without_product_type():
    record = _english_record(product_type=[])
    assert build_canonical_product(record) is None


def test_build_canonical_product_handles_missing_optional_fields():
    record = _english_record()
    for key in ("brand", "color", "material", "bullet_point", "item_keywords", "main_image_id"):
        record.pop(key, None)
    product = build_canonical_product(record)
    assert product is not None
    assert product.brand is None
    assert product.color is None
    assert product.material is None
    assert product.bullet_points == []
    assert product.keywords == []
    assert product.image_path is None


def test_build_canonical_product_is_deterministic():
    record = _english_record()
    p1 = build_canonical_product(record, image_index={"IMG123": "ab/abc123.jpg"})
    p2 = build_canonical_product(record, image_index={"IMG123": "ab/abc123.jpg"})
    assert p1.doc_text == p2.doc_text
    assert p1 == p2


# ---------------------------------------------------------------------------
# capped_stratified_sample
# ---------------------------------------------------------------------------


def _category_df(counts: dict[str, int]) -> pd.DataFrame:
    rows = [{"product_type": cat, "item_id": f"{cat}-{i}"} for cat, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows)


def test_capped_stratified_sample_caps_dominant_category():
    # One category at 80%, mirroring the real CELLULAR_PHONE_CASE skew (~50%+ of English subset).
    df = _category_df({"PHONE_CASE": 8000, "SHOES": 1000, "TABLE": 1000})
    sampled = capped_stratified_sample(df, target_size=1000, max_category_fraction=0.10, seed=42)

    counts = sampled["product_type"].value_counts()
    assert len(sampled) <= 1000
    assert counts["PHONE_CASE"] <= 100  # capped at 10% of target
    # Smaller categories end up proportionally better represented than in the raw catalog.
    assert counts["SHOES"] / len(sampled) > 1000 / len(df)
    assert counts["TABLE"] / len(sampled) > 1000 / len(df)


def test_capped_stratified_sample_never_starves_small_categories():
    df = _category_df({"PHONE_CASE": 8000, "RARE_CATEGORY": 5})
    sampled = capped_stratified_sample(df, target_size=1000, seed=42)
    assert (sampled["product_type"] == "RARE_CATEGORY").sum() == 5


def test_capped_stratified_sample_is_deterministic():
    df = _category_df({"A": 500, "B": 500, "C": 500})
    s1 = capped_stratified_sample(df, target_size=300, seed=7)
    s2 = capped_stratified_sample(df, target_size=300, seed=7)
    assert list(s1["item_id"]) == list(s2["item_id"])


def test_capped_stratified_sample_returns_all_rows_when_target_exceeds_size():
    df = _category_df({"A": 10, "B": 10})
    sampled = capped_stratified_sample(df, target_size=1000, seed=1)
    assert len(sampled) == len(df)


@pytest.mark.parametrize("max_fraction", [0.05, 0.10, 0.25])
def test_capped_stratified_sample_respects_various_caps(max_fraction):
    df = _category_df({"DOMINANT": 9000, "OTHER": 1000})
    sampled = capped_stratified_sample(df, target_size=1000, max_category_fraction=max_fraction, seed=3)
    cap = int(max_fraction * 1000)
    assert (sampled["product_type"] == "DOMINANT").sum() <= cap
