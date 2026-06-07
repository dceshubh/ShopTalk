"""Exit-gate suite for the data acquisition / EDA / preprocessing stage.

These run against the persisted sample (`data/processed/products.parquet`) produced by
`src.preprocess.build.build_products_dataset`, and against the raw catalog directly where a
check needs ground truth the sample alone can't provide (e.g. the schema audit). Skipped
automatically if the raw data hasn't been downloaded — this suite documents the *contract*
the pipeline must satisfy, independent of whether a given machine has the dataset materialized.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from src.common.config import load_config, resolve_path
from src.preprocess.build import build_canonical_dataframe, build_products_dataset
from src.preprocess.load import LISTINGS_GLOB

cfg = load_config()
RAW_DIR = resolve_path(cfg["paths"]["raw_dir"])
PRODUCTS_PARQUET = resolve_path(cfg["paths"]["products_parquet"])

raw_data_available = pytest.mark.skipif(
    not list(RAW_DIR.glob(LISTINGS_GLOB)),
    reason="Raw ABO listing shards not present — run the data download step first.",
)
sample_available = pytest.mark.skipif(
    not PRODUCTS_PARQUET.exists(),
    reason="data/processed/products.parquet not built — run `python -m src.preprocess.build`.",
)


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_parquet(PRODUCTS_PARQUET)


# ---------------------------------------------------------------------------
# Schema audit — settles the price / catalog-mix assumptions flagged pre-EDA
# ---------------------------------------------------------------------------


@raw_data_available
def test_schema_audit_confirms_no_price_field():
    """Settles the plan's flagged assumption: ABO has NO price field anywhere.

    This was recorded as a "high-confidence inference, must confirm" risk before EDA.
    EDA confirmed it definitively (0 / 147,702 records). The fallback chosen — drop price
    filtering, demo on color/material/type/dimensions instead of synthesizing fake prices —
    is documented in notebooks/01_eda.ipynb §1 and docs/ShopTalk_Plan.md.
    """
    price_like_count = 0
    total = 0
    for shard_path in sorted(RAW_DIR.glob(LISTINGS_GLOB)):
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                total += 1
                record = json.loads(line)
                if any(k in record for k in ("price", "list_price", "msrp")):
                    price_like_count += 1

    assert total > 100_000
    assert price_like_count == 0


@raw_data_available
def test_catalog_mix_matches_documented_eda_finding():
    """The pre-EDA "furniture-heavy" assumption was wrong; the real skew is phone cases
    (~44% of the raw catalog / ~53% of the English-typed canonical catalog). This assertion
    locks in that documented finding so a future re-run of the pipeline (e.g. against an
    updated dataset snapshot) surfaces a loud failure if the catalog's shape has
    fundamentally changed — rather than silently invalidating every demo query and
    golden-set theme built on top of it.

    Deliberately checked against the *full canonical catalog*, not the sampled
    `products_df`: `capped_stratified_sample` exists specifically to flatten this dominance
    down to `max_category_fraction` of the target size, so multiple categories legitimately
    tie at the cap in the sample — checking "is phone-case #1" there would be asserting
    against pandas' arbitrary tie-break order, not the actual EDA finding.
    """
    full_df = build_canonical_dataframe(RAW_DIR)
    top_category = full_df["product_type"].value_counts().index[0]
    assert top_category == "CELLULAR_PHONE_CASE"


# ---------------------------------------------------------------------------
# Determinism — "same input -> byte-identical output"
# ---------------------------------------------------------------------------


@raw_data_available
def test_build_canonical_dataframe_is_deterministic():
    df1 = build_canonical_dataframe(RAW_DIR)
    df2 = build_canonical_dataframe(RAW_DIR)
    pd.testing.assert_series_equal(
        df1["doc_text"].reset_index(drop=True),
        df2["doc_text"].reset_index(drop=True),
    )
    assert hash(tuple(df1["doc_text"])) == hash(tuple(df2["doc_text"]))


@raw_data_available
def test_build_products_dataset_is_reproducible(tmp_path):
    out1 = tmp_path / "products_1.parquet"
    out2 = tmp_path / "products_2.parquet"
    df1 = build_products_dataset(raw_dir=RAW_DIR, output_path=out1, target_size=500, seed=42)
    df2 = build_products_dataset(raw_dir=RAW_DIR, output_path=out2, target_size=500, seed=42)
    assert list(df1["item_id"]) == list(df2["item_id"])
    assert list(df1["doc_text"]) == list(df2["doc_text"])


# ---------------------------------------------------------------------------
# English-leakage check — "0% non-English in a manual sample"
# ---------------------------------------------------------------------------


@sample_available
def test_no_non_english_leakage_in_sample(products_df):
    """Every canonical `name` must come from an English-tagged source field.

    We can't re-derive "was this English-tagged at the source" from the parquet alone
    (the canonical doc only stores the selected value), so this check re-runs the
    selection logic's *predicate* against the exact raw records the canonical sample was
    built from, and confirms each one had >=1 English tag — i.e. confirms no record
    without an English `item_name` slipped through the filter.

    Crucially, ABO's uniqueness key is `(item_id, domain_name)`, not `item_id` alone (the
    same item_id can be cross-listed on multiple marketplaces with different language
    tagging — see `_dedupe_cross_marketplace_listings`). Matching on `item_id` alone could
    pull a *different* marketplace's non-English listing for the same id and report a false
    leak, so we match on the full `(item_id, domain_name)` pair the pipeline actually chose.
    """
    sample_pairs = set(
        products_df[["item_id", "domain_name"]]
        .sample(min(200, len(products_df)), random_state=42)
        .itertuples(index=False, name=None)
    )
    checked = 0
    leaked = 0

    for shard_path in sorted(RAW_DIR.glob(LISTINGS_GLOB)):
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                key = (record.get("item_id"), record.get("domain_name"))
                if key not in sample_pairs:
                    continue
                names = record.get("item_name", [])
                has_english = any(e.get("language_tag", "").startswith("en") for e in names)
                checked += 1
                if not has_english:
                    leaked += 1
        if checked >= len(sample_pairs):
            break

    assert checked > 0
    assert leaked == 0


# ---------------------------------------------------------------------------
# Coverage — "every selected product has a non-empty doc + resolvable image path"
# ---------------------------------------------------------------------------


@sample_available
def test_every_sampled_product_has_a_nonempty_doc(products_df):
    assert products_df["doc_text"].str.strip().str.len().gt(0).all()


@sample_available
def test_every_sampled_product_has_a_resolvable_image_path(products_df):
    resolvable = products_df["image_path"].notna()
    assert resolvable.all(), (
        f"{(~resolvable).sum()} / {len(products_df)} sampled products have no resolvable "
        "image path — captioning coverage would be incomplete."
    )


@sample_available
def test_no_duplicate_item_ids_in_sample(products_df):
    assert not products_df["item_id"].duplicated().any()


# ---------------------------------------------------------------------------
# Sampling contract — capped-stratified shape
# ---------------------------------------------------------------------------


@sample_available
def test_sample_respects_category_cap(products_df):
    cfg_local = load_config()
    target = cfg_local["dataset"]["subsample_size"]
    cap = int(0.10 * target)  # default max_category_fraction in capped_stratified_sample
    counts = products_df["product_type"].value_counts()
    assert counts.max() <= cap


@sample_available
def test_sample_size_is_close_to_target(products_df):
    target = load_config()["dataset"]["subsample_size"]
    assert len(products_df) <= target
    assert len(products_df) >= 0.9 * target
