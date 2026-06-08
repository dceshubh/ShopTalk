"""Unit tests for src.eval.harness — the Precision@K / MRR / Recall@K / NDCG metric
functions (pure, exercised against synthetic ranked lists — no model or index involved)
plus the orchestration glue (`evaluate`, `assert_filter_excludes_mismatches`,
`load_golden_set`) with retrieval faked. The numbers these functions produce are the
entire basis for "best encoder chosen with numbers, not vibes" — getting Precision@k's
denominator or MRR's rank-indexing off by one would silently invalidate that comparison.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.eval.harness import (
    GoldenCase,
    assert_filter_excludes_mismatches,
    evaluate,
    load_golden_set,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RELEVANT = frozenset({"A", "C"})

# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_counts_relevant_hits_in_the_top_k_slice():
    retrieved = ["A", "B", "C", "D", "E"]
    assert precision_at_k(retrieved, RELEVANT, k=1) == 1.0  # A is relevant
    assert precision_at_k(retrieved, RELEVANT, k=2) == 0.5  # A relevant, B not
    assert precision_at_k(retrieved, RELEVANT, k=5) == pytest.approx(2 / 5)  # A and C


def test_precision_at_k_divides_by_k_even_when_fewer_results_are_returned():
    # Returning 1 result for k=5 is itself a miss on the other 4 slots — the
    # denominator must stay `k`, not `len(retrieved)`, or a thin result set looks "perfect".
    assert precision_at_k(["A"], RELEVANT, k=5) == pytest.approx(1 / 5)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_is_one_over_the_rank_of_the_first_relevant_hit():
    assert reciprocal_rank(["A", "B", "C"], RELEVANT) == 1.0
    assert reciprocal_rank(["B", "A", "C"], RELEVANT) == 0.5
    assert reciprocal_rank(["B", "D", "C"], RELEVANT) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved():
    assert reciprocal_rank(["B", "D", "E"], RELEVANT) == 0.0
    assert reciprocal_rank([], RELEVANT) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_is_fraction_of_known_relevant_items_surfaced():
    # RELEVANT has 2 members (A, C); top-2 surfaces only A.
    assert recall_at_k(["A", "B", "C"], RELEVANT, k=1) == 0.5
    assert recall_at_k(["A", "B", "C"], RELEVANT, k=3) == 1.0


def test_recall_at_k_is_zero_for_an_empty_relevance_set_not_a_division_error():
    assert recall_at_k(["A", "B"], frozenset(), k=5) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_at_k_is_one_when_relevant_items_occupy_the_top_ranks():
    # Two relevant items (A, C); placing them in ranks 1-2 is the best possible
    # arrangement for k=2 -> DCG == IDCG -> NDCG == 1.0.
    assert ndcg_at_k(["A", "C", "B"], RELEVANT, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_relevant_items_appearing_lower_in_the_ranking():
    perfect = ndcg_at_k(["A", "C", "B"], RELEVANT, k=3)
    worse = ndcg_at_k(["B", "A", "C"], RELEVANT, k=3)
    assert worse < perfect
    assert 0.0 < worse < 1.0


def test_ndcg_at_k_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(["B", "D", "E"], RELEVANT, k=3) == 0.0


# ---------------------------------------------------------------------------
# load_golden_set
# ---------------------------------------------------------------------------


def test_load_golden_set_parses_cases_into_frozensets(tmp_path):
    golden_path = tmp_path / "golden_set.json"
    golden_path.write_text(
        '{"_meta": {}, "cases": ['
        '{"query": "a blue chair", "relevant_item_ids": ["P1", "P2"], "category": "attribute"},'
        '{"query": "a red sofa", "relevant_item_ids": ["P3"], "category": "easy"}'
        "]}"
    )

    cases = load_golden_set(golden_path)

    assert cases == [
        GoldenCase(query="a blue chair", relevant_item_ids=frozenset({"P1", "P2"}), category="attribute"),
        GoldenCase(query="a red sofa", relevant_item_ids=frozenset({"P3"}), category="easy"),
    ]


# ---------------------------------------------------------------------------
# evaluate — orchestration glue, retrieval faked via `search`
# ---------------------------------------------------------------------------


def test_evaluate_averages_metrics_across_cases_and_runs_one_search_per_query():
    cases = [
        GoldenCase("query one", frozenset({"A"}), "easy"),
        GoldenCase("query two", frozenset({"Z"}), "easy"),
    ]
    # Query one: "A" at rank 1 -> perfect. Query two: "Z" not retrieved -> all zeros.
    with patch("src.eval.harness.search", side_effect=[["A", "B", "C"], ["B", "C", "D"]]) as mock_search:
        metrics = evaluate(MagicMock(), MagicMock(), cases, k_values=[1, 3])

    assert mock_search.call_count == 2
    assert metrics["mrr"] == pytest.approx((1.0 + 0.0) / 2)
    assert metrics["precision_at_1"] == pytest.approx((1.0 + 0.0) / 2)
    assert metrics["precision_at_3"] == pytest.approx(((1 / 3) + 0.0) / 2)
    assert metrics["recall_at_3"] == pytest.approx((1.0 + 0.0) / 2)


def test_evaluate_requests_top_k_equal_to_the_largest_k_value_only_once_per_query():
    cases = [GoldenCase("query", frozenset({"A"}), "easy")]
    with patch("src.eval.harness.search", return_value=["A"]) as mock_search:
        evaluate(MagicMock(), MagicMock(), cases, k_values=[1, 5, 10])

    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs["top_k"] == 10


# ---------------------------------------------------------------------------
# assert_filter_excludes_mismatches
# ---------------------------------------------------------------------------


def test_assert_filter_excludes_mismatches_passes_when_every_result_matches():
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["P1", "P2"],
        "metadatas": [
            {"color": "Blue", "product_type": "CHAIR"},
            {"color": "Blue", "product_type": "CHAIR"},
        ],
    }
    case = GoldenCase("a blue chair", frozenset({"P1"}), "attribute")
    where = {"$and": [{"color": "Blue"}, {"product_type": "CHAIR"}]}

    with patch("src.eval.harness.search", return_value=["P1", "P2"]):
        assert_filter_excludes_mismatches(collection, MagicMock(), case, where)  # no raise


def test_assert_filter_excludes_mismatches_raises_on_a_mismatched_attribute():
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["P1", "P2"],
        "metadatas": [
            {"color": "Blue", "product_type": "CHAIR"},
            {"color": "Red", "product_type": "CHAIR"},  # the violation: "blue chair" -> red item
        ],
    }
    case = GoldenCase("a blue chair", frozenset({"P1"}), "attribute")
    where = {"$and": [{"color": "Blue"}, {"product_type": "CHAIR"}]}

    with patch("src.eval.harness.search", return_value=["P1", "P2"]):
        with pytest.raises(AssertionError, match="Structured filter violated"):
            assert_filter_excludes_mismatches(collection, MagicMock(), case, where)


def test_assert_filter_excludes_mismatches_short_circuits_on_an_empty_result_set():
    collection = MagicMock()
    case = GoldenCase("a paisley unicorn saddle", frozenset({"P1"}), "hard")

    with patch("src.eval.harness.search", return_value=[]):
        assert_filter_excludes_mismatches(collection, MagicMock(), case, {"color": "Blue"})

    collection.get.assert_not_called()
