"""Unit tests for src.ui.feedback — the SQLite-backed 👍/👎 store behind Phase 7's feedback
buttons and Phase 9's hard-negative aggregation. Uses a real temp-file SQLite DB (sqlite3 is
stdlib — no value in mocking the one thing we're actually testing: that the upsert/query SQL
is correct against the real engine)."""

from __future__ import annotations

import pytest

from src.ui.feedback import FeedbackStore, load_feedback_store


@pytest.fixture
def store(tmp_path):
    return load_feedback_store(tmp_path / "feedback.sqlite")


def test_record_persists_a_retrievable_verdict(store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="up")

    assert store.verdict_for(user_id="u1", query="brown chairs", item_id="P1") == "up"


def test_verdict_for_returns_none_when_no_feedback_was_given(store: FeedbackStore):
    assert store.verdict_for(user_id="u1", query="brown chairs", item_id="P1") is None


def test_re_recording_the_opposite_verdict_overwrites_rather_than_duplicates(store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="up")
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="down")

    assert store.verdict_for(user_id="u1", query="brown chairs", item_id="P1") == "down"
    assert len(store.all_with_verdict("up")) == 0
    assert len(store.all_with_verdict("down")) == 1


def test_distinct_users_or_queries_or_products_are_independent_records(store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="up")
    store.record(user_id="u2", session_id="s2", query="brown chairs", item_id="P1", verdict="down")
    store.record(user_id="u1", session_id="s1", query="blue rugs", item_id="P1", verdict="down")
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P2", verdict="down")

    assert store.verdict_for(user_id="u1", query="brown chairs", item_id="P1") == "up"
    assert store.verdict_for(user_id="u2", query="brown chairs", item_id="P1") == "down"
    assert len(store.all_with_verdict("down")) == 3


def test_all_with_verdict_returns_full_records_for_aggregation(store: FeedbackStore):
    store.record(user_id="u1", session_id="s1", query="brown chairs", item_id="P1", verdict="down")

    [record] = store.all_with_verdict("down")
    assert record["user_id"] == "u1"
    assert record["session_id"] == "s1"
    assert record["query"] == "brown chairs"
    assert record["item_id"] == "P1"
    assert record["verdict"] == "down"
    assert "ts" in record


def test_record_rejects_an_invalid_verdict(store: FeedbackStore):
    with pytest.raises(ValueError, match="verdict must be 'up' or 'down'"):
        store.record(user_id="u1", session_id="s1", query="q", item_id="P1", verdict="meh")


def test_load_feedback_store_creates_the_parent_directory_and_schema(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "feedback.sqlite"

    store = load_feedback_store(db_path)
    store.record(user_id="u1", session_id="s1", query="q", item_id="P1", verdict="up")

    assert db_path.exists()
    assert store.verdict_for(user_id="u1", query="q", item_id="P1") == "up"
