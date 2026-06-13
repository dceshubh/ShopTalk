"""Retrieval eval harness — runs the hand-written golden set (`data/eval/golden_set.json`)
against a Chroma collection and reports Precision@K / MRR (primary, per `eval.metrics_primary`
in config.yaml) plus Recall@K / NDCG (approximate — a ~50-case sampled set can't give
exhaustive or graded relevance labels across a 40K-product catalog, so these are reported
as directional, not authoritative).

This module is also where the comparison-sweep table (3 encoders x {text-only,
caption-enriched}) gets produced — the artifact that turns "we picked bge-base" into
"we picked bge-base, and here are the numbers that show why" (docs/ShopTalk_Plan.md
Phase-3 exit gate: "best pretrained encoder chosen with numbers, not vibes").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.config import load_config, resolve_path
from src.common.logging import get_logger
from src.embeddings.encode import Encoder, load_encoder
from src.index.build import CorpusType, build_collection, search

logger = get_logger(__name__)


@dataclass(frozen=True)
class GoldenCase:
    query: str
    relevant_item_ids: frozenset[str]
    category: str


def load_golden_set(path: Path | str | None = None) -> list[GoldenCase]:
    """Load the hand-written (query -> relevant product_ids) cases. See `_meta` in the
    JSON file itself for methodology, scope, and the eval-integrity rationale."""
    path = Path(path) if path else resolve_path(load_config()["paths"]["golden_set"])
    with open(path) as f:
        raw = json.load(f)
    return [
        GoldenCase(
            query=case["query"],
            relevant_item_ids=frozenset(case["relevant_item_ids"]),
            category=case["category"],
        )
        for case in raw["cases"]
    ]


# ---------------------------------------------------------------------------
# Metric functions — pure, take a ranked id list + a relevance set, return a float.
# Kept separate from any retrieval call so they're trivial to unit-test with synthetic
# ranked lists (tests/test_harness.py) and reusable for the fine-tune-vs-pretrained
# comparison in Phase 4 without re-running retrieval.
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the top-`k` results that are relevant. Always divides by `k` (not by
    the number actually retrieved) — returning fewer than `k` results is itself a miss."""
    return sum(1 for item_id in retrieved[:k] if item_id in relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: frozenset[str]) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none appears. MRR is the mean of
    this across queries — rewards "the right answer near the top", which for a chat-style
    shopping assistant (one item surfaced and discussed at a time) matters more than
    "all relevant items somewhere in the top 10"."""
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of ALL known-relevant items that appear in the top-`k`. Reported as
    APPROXIMATE: `relevant` here is a hand-picked sample, not an exhaustive label set, so
    a "miss" may really be an unlabeled-but-valid match the harness can't see."""
    if not relevant:
        return 0.0
    return sum(1 for item_id in retrieved[:k] if item_id in relevant) / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Binary-relevance NDCG@k (graded relevance would need human 0-3 scores per item,
    which this golden set doesn't have — see `recall_at_k` docstring; same caveat applies)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(retrieved[:k], start=1)
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


_METRIC_FNS = {
    "precision_at_k": precision_at_k,
    "recall_at_k": recall_at_k,
    "ndcg_at_k": ndcg_at_k,
}


def evaluate(
    collection,
    encoder: Encoder,
    cases: list[GoldenCase],
    *,
    k_values: list[int],
) -> dict[str, float]:
    """Run every golden query through `collection` (embedded with `encoder` — the same
    encoder the collection was indexed with, preserving "same transformers train<->inference")
    and return the mean of each metric across all cases.

    Retrieves `top_k = max(k_values)` once per query and slices it for every smaller `k` —
    one ANN search per query, not one per (query, k) pair.
    """
    if not cases:
        return {}

    top_k = max(k_values)
    per_case: list[dict[str, float]] = []

    for case in cases:
        retrieved = search(collection, encoder, case.query, top_k=top_k)
        row: dict[str, float] = {"mrr": reciprocal_rank(retrieved, case.relevant_item_ids)}
        for k in k_values:
            for metric_name, fn in _METRIC_FNS.items():
                row[f"{metric_name.replace('_k', f'_{k}')}"] = fn(retrieved, case.relevant_item_ids, k)
        per_case.append(row)

    metric_names = per_case[0].keys()
    return {name: sum(row[name] for row in per_case) / len(per_case) for name in metric_names}


# ---------------------------------------------------------------------------
# Structured-filter correctness — the Phase-3 exit gate "a 'blue chair' query never
# returns a red item or a non-chair in top-K", checked against real golden cases tagged
# `category: attribute` (each names an exact color/type combination present in the corpus).
# ---------------------------------------------------------------------------


def assert_filter_excludes_mismatches(
    collection,
    encoder: Encoder,
    case: GoldenCase,
    where: dict,
    *,
    top_k: int = 10,
) -> None:
    """Run `case.query` through `collection` WITH the structured pre-filter `where` applied,
    and assert every returned item's metadata satisfies it — proof that "SQL-then-semantic"
    is doing real filtering, not just nudging rank order. Raises `AssertionError` on the
    first mismatch (this is a correctness gate, not a soft metric).
    """
    retrieved_ids = search(collection, encoder, case.query, top_k=top_k, where=where)
    if not retrieved_ids:
        return
    got = collection.get(ids=retrieved_ids, include=["metadatas"])
    for item_id, metadata in zip(got["ids"], got["metadatas"]):
        for key, expected in _flatten_where(where).items():
            actual = metadata.get(key)
            assert actual == expected, (
                f"Structured filter violated for query {case.query!r}: "
                f"item {item_id} has {key}={actual!r}, expected {expected!r} (where={where})"
            )


def _flatten_where(where: dict) -> dict[str, str]:
    """Flatten a Chroma `{"$and": [{"color": "Blue"}, {"product_type": "CHAIR"}]}` filter
    into `{"color": "Blue", "product_type": "CHAIR"}` for per-field assertion. Only the
    `$and`-of-equality shape this harness constructs needs to be supported here."""
    if "$and" in where:
        flat: dict[str, str] = {}
        for clause in where["$and"]:
            flat.update(clause)
        return flat
    return dict(where)


# ---------------------------------------------------------------------------
# Comparison sweep — build (encoder x corpus) collections and print the metric table that
# justifies "best pretrained encoder chosen with numbers, not vibes".
# ---------------------------------------------------------------------------


def run_comparison_sweep(
    *,
    enriched_path: Path | str | None = None,
    chroma_dir: Path | str | None = None,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Build a Chroma collection for every (encoder, corpus_type) cell, evaluate each
    against the golden set, and return/print a comparison table.

    `enriched_path` defaults to the dev-scale (200-doc) BLIP-2-captioned sample — the
    "dev-scale now, full-scale later" scoping documented in PROJECT_REPORT.md §3.3. Re-run
    with the full-catalog enriched parquet once the full BLIP-2 batch lands; nothing else
    in this function changes.
    """
    cfg = load_config()
    enriched_path = (
        Path(enriched_path)
        if enriched_path
        else resolve_path(cfg["paths"]["data_dir"])
        / "kaggle_process"
        / "products_enriched_blip2-opt-2.7b.parquet"
    )
    k_values = k_values or cfg["eval"]["k_values"]

    df = pd.read_parquet(enriched_path)
    cases = load_golden_set()
    encoder_cfg = cfg["models"]["text_encoder"]
    model_names = [encoder_cfg["primary"], *encoder_cfg["compare"]]
    corpus_types: list[CorpusType] = ["text_only", "caption_enriched"]

    rows = []
    for model_name in model_names:
        encoder = load_encoder(model_name)
        for corpus_type in corpus_types:
            collection = build_collection(df, encoder, corpus_type, chroma_dir=chroma_dir)
            metrics = evaluate(collection, encoder, cases, k_values=k_values)
            rows.append({"model": model_name, "corpus": corpus_type, **metrics})
            logger.info("Evaluated %s / %s: %s", model_name, corpus_type, metrics)

    table = pd.DataFrame(rows)
    logger.info("\n%s", table.to_string(index=False))
    return table


if __name__ == "__main__":
    run_comparison_sweep()
