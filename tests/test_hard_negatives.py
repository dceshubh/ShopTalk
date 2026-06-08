"""Unit tests for src.eval.hard_negatives — re-mining 👎/👍 feedback into hard-negative
training triplets for the next fine-tuning round (docs/ShopTalk_Plan.md Phase 9 exit gate:
"re-mining from feedback produces valid training triplets"). Driven against a REAL
`FeedbackStore` (SQLite, stdlib) — same "no value mocking the thing under test" rationale
as tests/test_feedback.py.
"""

from __future__ import annotations

import pytest

from src.eval.hard_negatives import HardNegativeTriplet, mine_hard_negatives
from src.ui.feedback import FeedbackStore, load_feedback_store


@pytest.fixture
def store(tmp_path) -> FeedbackStore:
    return load_feedback_store(tmp_path / "feedback.sqlite")


def test_a_same_user_same_query_up_and_down_pair_becomes_one_valid_triplet(store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="LIKED1", verdict="up")
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="DISLIKED1", verdict="down")

    [triplet] = mine_hard_negatives(store)

    assert triplet == HardNegativeTriplet(
        query="brown chairs", positive_item_id="LIKED1", negative_item_id="DISLIKED1"
    )


def test_a_downvote_with_no_matching_same_query_upvote_yields_no_triplet(store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="DISLIKED1", verdict="down")
    store.record(user_id="u1", session_id="s1", query="blue rugs", item_id="LIKED1", verdict="up")

    assert mine_hard_negatives(store) == []


def test_cross_joins_every_matching_pair_for_one_query_not_just_the_first(store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="LIKED1", verdict="up")
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="LIKED2", verdict="up")
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="DISLIKED1", verdict="down")

    triplets = mine_hard_negatives(store)

    assert len(triplets) == 2
    assert {t.positive_item_id for t in triplets} == {"LIKED1", "LIKED2"}
    assert all(t.negative_item_id == "DISLIKED1" and t.query == "brown chairs" for t in triplets)


def test_pairs_are_scoped_to_one_user_not_shared_across_users(store: FeedbackStore):
    """A 👎 from u1 must never be paired with u2's 👍 — that would teach the encoder a
    relevance judgment neither user actually made."""
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="DISLIKED1", verdict="down")
    store.record(user_id="u2", session_id="s2", query="brown chairs", item_id="LIKED1", verdict="up")

    assert mine_hard_negatives(store) == []


def test_every_triplet_field_is_a_non_empty_string(store: FeedbackStore):
    """The Phase-9 exit gate, verbatim: "valid training triplets" — each leg must be a
    real, usable string an embedding pipeline can resolve to doc text and encode."""
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="LIKED1", verdict="up")
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="DISLIKED1", verdict="down")

    [triplet] = mine_hard_negatives(store)

    for field_value in (triplet.query, triplet.positive_item_id, triplet.negative_item_id):
        assert isinstance(field_value, str) and field_value
    assert triplet.positive_item_id != triplet.negative_item_id
