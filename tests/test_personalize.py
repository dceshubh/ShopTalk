"""Unit tests for src.agent.personalize — the Phase-9 feedback-driven re-ranker.

Exercises `Personalizer.rerank` against a REAL `FeedbackStore` (SQLite, stdlib — same "no
value in mocking the one thing under test" rationale as tests/test_feedback.py) with fake
metadata and preferences, proving the three Phase-9 exit-gate properties directly:
"a 👎'd product is demonstrably down-ranked," "personalized vs non-personalized results
differ for a user with history," and (implicitly) "no history -> unchanged ranking."
"""

from __future__ import annotations

import pytest

from src.agent.memory import UserPreferences
from src.agent.personalize import Personalizer
from src.ui.feedback import FeedbackStore, load_feedback_store


@pytest.fixture
def store(tmp_path) -> FeedbackStore:
    return load_feedback_store(tmp_path / "feedback.sqlite")


@pytest.fixture
def personalizer(store: FeedbackStore) -> Personalizer:
    return Personalizer(feedback_store=store)


_METADATA = {
    "P1": {"product_type": "CHAIR", "color": "Brown"},
    "P2": {"product_type": "CHAIR", "color": "Black"},
    "P3": {"product_type": "CHAIR", "color": "Beige"},
    "P4": {"product_type": "CHAIR", "color": "Brown"},
    "P5": {"product_type": "CHAIR", "color": "Grey"},
}


def test_a_user_with_no_history_gets_back_the_similarity_order_unchanged(personalizer: Personalizer):
    candidate_ids = ["P1", "P2", "P3", "P4", "P5"]

    result = personalizer.rerank(
        candidate_ids,
        user_id="fresh-user",
        preferences=UserPreferences(),
        metadata_by_id=_METADATA,
        top_k=3,
    )

    assert result == ["P1", "P2", "P3"]


def test_a_downvoted_product_is_demonstrably_down_ranked_for_that_user(personalizer, store: FeedbackStore):
    """The Phase-9 exit gate, verbatim: "a 👎'd product is demonstrably down-ranked for
    that user on the next similar query." P1 ranks #1 by raw similarity but this user
    explicitly said "not for me" — it must not resurface near the top for them."""
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="down")
    candidate_ids = ["P1", "P2", "P3", "P4", "P5"]

    result = personalizer.rerank(
        candidate_ids, user_id="u1", preferences=UserPreferences(), metadata_by_id=_METADATA, top_k=5
    )

    assert result[-1] == "P1"  # pushed to the very bottom of its own candidate pool
    assert "P1" not in result[:3]


def test_downvotes_are_scoped_to_the_user_who_gave_them(personalizer, store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="down")
    candidate_ids = ["P1", "P2", "P3"]

    result_for_u1 = personalizer.rerank(
        candidate_ids, user_id="u1", preferences=UserPreferences(), metadata_by_id=_METADATA, top_k=3
    )
    result_for_u2 = personalizer.rerank(
        candidate_ids, user_id="u2", preferences=UserPreferences(), metadata_by_id=_METADATA, top_k=3
    )

    assert result_for_u1 != result_for_u2
    assert result_for_u1[-1] == "P1"
    assert result_for_u2 == ["P1", "P2", "P3"]  # u2 has no history — unaffected


def test_a_previously_liked_product_is_boosted_back_toward_the_top(personalizer, store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="chairs", item_id="P5", verdict="up")
    candidate_ids = ["P1", "P2", "P3", "P4", "P5"]  # P5 ranks last by raw similarity

    result = personalizer.rerank(
        candidate_ids, user_id="u1", preferences=UserPreferences(), metadata_by_id=_METADATA, top_k=5
    )

    assert result.index("P5") < candidate_ids.index("P5")


def test_preferred_colors_boost_matching_candidates(personalizer):
    candidate_ids = ["P2", "P5", "P1"]  # raw order: Black, Grey, Brown

    result = personalizer.rerank(
        candidate_ids,
        user_id="u1",
        preferences=UserPreferences(preferred_colors=["Brown"]),
        metadata_by_id=_METADATA,
        top_k=3,
    )

    assert result[0] == "P1"  # the Brown chair jumps to the front on a color-preference boost


def test_personalized_and_unpersonalized_rankings_differ_for_a_user_with_history(
    personalizer, store: FeedbackStore
):
    """The Phase-9 exit gate, verbatim: "personalized vs non-personalized results differ
    for a user with history (documented)." `candidate_ids[:top_k]` IS the unpersonalized
    ranking `_search_products_node` would return without a `Personalizer` in the loop."""
    candidate_ids = ["P1", "P2", "P3", "P4", "P5"]
    unpersonalized = candidate_ids[:3]

    store.record(user_id="u1", session_id="s1", query="chairs", item_id="P1", verdict="down")
    store.record(user_id="u1", session_id="s1", query="chairs", item_id="P5", verdict="up")

    personalized = personalizer.rerank(
        candidate_ids, user_id="u1", preferences=UserPreferences(), metadata_by_id=_METADATA, top_k=3
    )

    assert personalized != unpersonalized


def test_a_single_downvote_outweighs_any_combination_of_boosts(personalizer, store: FeedbackStore):
    """A "not for me" signal must never be out-voted by a re-surfacing or color match —
    otherwise clicking 👎 on a product the user keeps liking-adjacent items to would feel
    like it does nothing."""
    # Two DISTINCT (query, item_id) rows — the store's `UNIQUE (user_id, query, item_id)`
    # upsert would collapse same-query re-votes into one, which isn't the scenario here:
    # this user downvoted P1 on one search and later, separately, liked it on another.
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="down")
    store.record(user_id="u1", session_id="s1", query="recliners", item_id="P1", verdict="up")
    candidate_ids = ["P1", "P2"]

    result = personalizer.rerank(
        candidate_ids,
        user_id="u1",
        preferences=UserPreferences(preferred_colors=["Brown"]),  # P1 is Brown
        metadata_by_id=_METADATA,
        top_k=2,
    )

    assert result == ["P2", "P1"]
