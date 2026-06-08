"""Unit tests for src.agent.filters — combined history-aware query rewriting + structured
filter extraction. The LLM is faked (a real call needs GROQ_API_KEY and costs money); what
we assert on is the part that's load-bearing for the Phase-5 exit gates: that
`extract_filters` wires conversation history + the current message into the LLM call in
the right shape, and that `filters_to_where` builds EXACTLY the Chroma `where` clause shape
`src.index.build.search` expects (single-clause vs. `$and`, `None` when nothing extracted).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.filters import SearchFilters, extract_filters, filters_to_where

# ---------------------------------------------------------------------------
# extract_filters
# ---------------------------------------------------------------------------


def test_extract_filters_passes_history_then_current_message_to_complete_structured():
    llm = MagicMock()
    llm.complete_structured.return_value = SearchFilters(rewritten_query="cheap wooden dining chairs")
    history = [
        {"role": "user", "content": "show me wooden dining chairs"},
        {"role": "assistant", "content": "Here are three wooden dining chairs."},
    ]

    result = extract_filters(llm, history, "show me cheaper ones")

    assert result.rewritten_query == "cheap wooden dining chairs"
    (messages, response_model), _ = llm.complete_structured.call_args
    assert response_model is SearchFilters
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history  # history forwarded in order, untouched
    assert messages[-1] == {"role": "user", "content": "show me cheaper ones"}


def test_extract_filters_works_with_empty_history_for_a_first_turn():
    llm = MagicMock()
    llm.complete_structured.return_value = SearchFilters(
        rewritten_query="blue chairs", color="Blue", product_type="CHAIR"
    )

    result = extract_filters(llm, [], "do you have any blue chairs?")

    assert result.color == "Blue"
    (messages, _), _ = llm.complete_structured.call_args
    assert len(messages) == 2  # system + current message only, no history turns
    assert messages[-1]["content"] == "do you have any blue chairs?"


# ---------------------------------------------------------------------------
# filters_to_where
# ---------------------------------------------------------------------------


def test_filters_to_where_returns_none_when_nothing_was_extracted():
    filters = SearchFilters(rewritten_query="something nice for my kitchen")

    assert filters_to_where(filters) is None


def test_filters_to_where_returns_a_bare_clause_for_a_single_attribute():
    filters = SearchFilters(rewritten_query="blue rug", color="Blue")

    assert filters_to_where(filters) == {"color": "Blue"}


def test_filters_to_where_returns_an_and_clause_for_multiple_attributes():
    filters = SearchFilters(rewritten_query="a blue chair", color="Blue", product_type="CHAIR")

    where = filters_to_where(filters)

    assert where == {"$and": [{"product_type": "CHAIR"}, {"color": "Blue"}]}


def test_filters_to_where_ignores_rewritten_query_and_unknown_fields():
    # rewritten_query is search TEXT, not a metadata filter — must never leak into `where`.
    filters = SearchFilters(rewritten_query="a brand x leather sofa", brand="BrandX", material="Leather")

    where = filters_to_where(filters)

    assert where == {"$and": [{"material": "Leather"}, {"brand": "BrandX"}]}
    assert "rewritten_query" not in str(where)


@pytest.mark.parametrize("column", ["product_type", "color", "material", "brand"])
def test_filters_to_where_handles_every_registered_metadata_column(column):
    filters = SearchFilters(rewritten_query="q", **{column: "Some Value"})

    assert filters_to_where(filters) == {column: "Some Value"}
