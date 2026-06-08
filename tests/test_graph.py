"""Unit tests for src.agent.graph — the LangGraph conversational pipeline and the
`ShoppingAgent` orchestration wrapper. The LLM, Chroma collection, and encoder are all
faked (a real run needs GROQ_API_KEY + a built index); what we assert on is the graph
WIRING and the structural anti-hallucination property the Phase-5 exit gate names:
"generated answers cite real retrieved product_ids, no hallucinated products."

`PersistentMemory` is exercised against the real local Redis (db=15, flushed per test) —
same rationale as tests/test_memory.py: a faked Redis can't prove the round-trip works.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import redis

from src.agent.filters import SearchFilters
from src.agent.graph import ShoppingAgent, _describe_products, build_graph
from src.agent.memory import PersistentMemory, UserPreferences

TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture
def persistent_memory():
    client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not running locally — `brew services start redis` to enable this test")
    client.flushdb()
    yield PersistentMemory(redis_client=client)
    client.flushdb()


def _fake_llm(*, filters: SearchFilters, response_text: str = "Here are some great picks!"):
    llm = MagicMock()
    llm.complete_structured.return_value = filters
    llm.complete.return_value = response_text
    return llm


def _fake_collection(*, get_return=None):
    collection = MagicMock()
    collection.get.return_value = get_return or {"ids": [], "documents": [], "metadatas": []}
    return collection


# ---------------------------------------------------------------------------
# _describe_products — the grounding text the generator is restricted to
# ---------------------------------------------------------------------------


def test_describe_products_returns_empty_for_no_results_without_calling_get():
    collection = _fake_collection()

    assert _describe_products(collection, []) == []
    collection.get.assert_not_called()


def test_describe_products_renders_id_document_and_metadata_per_line():
    collection = _fake_collection(
        get_return={
            "ids": ["B07HZ1RYNT"],
            "documents": ["a brown leather recliner chair with great comfort " + "x" * 300],
            "metadatas": [{"color": "Brown", "product_type": "CHAIR"}],
        }
    )

    lines = _describe_products(collection, ["B07HZ1RYNT"])

    assert len(lines) == 1
    assert lines[0].startswith("- [B07HZ1RYNT] a brown leather recliner chair")
    assert "color=Brown" in lines[0]
    assert "product_type=CHAIR" in lines[0]
    assert len(lines[0]) < 350  # document text is truncated, not dumped whole


# ---------------------------------------------------------------------------
# build_graph — node wiring
# ---------------------------------------------------------------------------


def test_build_graph_runs_understand_then_search_then_generate_in_order():
    filters = SearchFilters(rewritten_query="a brown leather chair", color="Brown", product_type="CHAIR")
    llm = _fake_llm(filters=filters, response_text="The brown recliner looks like a great fit!")
    collection = _fake_collection(
        get_return={"ids": ["P1"], "documents": ["a chair"], "metadatas": [{"color": "Brown"}]}
    )
    encoder = MagicMock()

    with patch("src.agent.graph.search", return_value=["P1", "P2"]) as mock_search:
        graph = build_graph(llm, collection, encoder, top_k=5)
        result = graph.invoke({"user_id": "u1", "history": [], "message": "brown leather chairs?"})

    # understand_query ran first and its output drove the search call
    llm.complete_structured.assert_called_once()
    mock_search.assert_called_once()
    _, search_kwargs = mock_search.call_args
    assert search_kwargs["top_k"] == 5
    assert search_kwargs["where"] == {"$and": [{"product_type": "CHAIR"}, {"color": "Brown"}]}

    # generate_response ran last and saw the search results
    assert result["retrieved_ids"] == ["P1", "P2"]
    assert result["response_text"] == "The brown recliner looks like a great fit!"
    assert result["filters"] == filters


def test_build_graph_passes_an_unfiltered_search_when_no_attributes_were_extracted():
    filters = SearchFilters(rewritten_query="something nice for my kitchen")
    llm = _fake_llm(filters=filters)
    collection = _fake_collection()
    encoder = MagicMock()

    with patch("src.agent.graph.search", return_value=[]) as mock_search:
        graph = build_graph(llm, collection, encoder)
        graph.invoke({"user_id": "u1", "history": [], "message": "something nice for my kitchen?"})

    _, search_kwargs = mock_search.call_args
    assert search_kwargs["where"] is None


# ---------------------------------------------------------------------------
# ShoppingAgent — session buffering, persistent-memory merge, anti-hallucination
# ---------------------------------------------------------------------------


def test_shopping_agent_product_ids_are_sourced_from_retrieval_not_from_llm_text():
    """The structural anti-hallucination property: AgentTurn.product_ids IS retrieved_ids
    — never independently parsed out of the LLM's free-text response. Even if the LLM's
    prose mentions a totally different (hallucinated) id, the surfaced product_ids are
    unaffected, because they never came from the text in the first place."""
    filters = SearchFilters(rewritten_query="blue rugs", color="Blue", product_type="RUG")
    fake_response_text = "I'd recommend item B00FAKE0001, a gorgeous rug!"  # not in retrieved_ids
    llm = _fake_llm(filters=filters, response_text=fake_response_text)
    collection = _fake_collection()
    encoder = MagicMock()

    with patch("src.agent.graph.search", return_value=["REAL1", "REAL2"]):
        agent = ShoppingAgent(graph=build_graph(llm, collection, encoder), persistent_memory=_noop_memory())
        turn = agent.chat(user_id="u1", session_id="s1", message="blue rugs?")

    assert turn.product_ids == ["REAL1", "REAL2"]
    assert "B00FAKE0001" not in turn.product_ids


def test_shopping_agent_carries_conversation_history_across_turns_in_one_session():
    filters = SearchFilters(rewritten_query="q")
    llm = _fake_llm(filters=filters, response_text="ok")
    collection = _fake_collection()
    encoder = MagicMock()

    with patch("src.agent.graph.search", return_value=[]):
        agent = ShoppingAgent(graph=build_graph(llm, collection, encoder), persistent_memory=_noop_memory())
        agent.chat(user_id="u1", session_id="s1", message="show me chairs")
        agent.chat(user_id="u1", session_id="s1", message="cheaper ones?")

    # second call's `history` must contain the first turn's user+assistant messages
    second_call_state = llm.complete_structured.call_args_list[1].args[0]
    history_messages = [m for m in second_call_state if m["role"] in ("user", "assistant")]
    assert {"role": "user", "content": "show me chairs"} in history_messages
    assert {"role": "assistant", "content": "ok"} in history_messages


def test_shopping_agent_keeps_separate_buffers_per_session():
    filters = SearchFilters(rewritten_query="q")
    llm = _fake_llm(filters=filters, response_text="ok")
    collection = _fake_collection()
    encoder = MagicMock()

    with patch("src.agent.graph.search", return_value=[]):
        agent = ShoppingAgent(graph=build_graph(llm, collection, encoder), persistent_memory=_noop_memory())
        agent.chat(user_id="u1", session_id="session-A", message="message in session A")
        agent.chat(user_id="u1", session_id="session-B", message="message in session B")

    assert "session-A" in agent._buffers
    assert "session-B" in agent._buffers
    assert agent._buffers["session-A"].messages[0]["content"] == "message in session A"
    assert agent._buffers["session-B"].messages[0]["content"] == "message in session B"


def test_shopping_agent_merges_extracted_color_into_persistent_preferences(persistent_memory):
    filters = SearchFilters(rewritten_query="green chairs", color="Green", product_type="CHAIR")
    llm = _fake_llm(filters=filters, response_text="ok")
    collection = _fake_collection()
    encoder = MagicMock()

    with patch("src.agent.graph.search", return_value=[]):
        agent = ShoppingAgent(
            graph=build_graph(llm, collection, encoder), persistent_memory=persistent_memory
        )
        agent.chat(user_id="user-green", session_id="s1", message="green chairs please")

    assert persistent_memory.load("user-green") == UserPreferences(preferred_colors=["Green"])


def test_shopping_agent_persisted_preference_survives_a_fresh_connection(persistent_memory):
    """End-to-end version of the Phase-5 exit gate "persistent pref survives a session
    restart": chat once (writes through `merge`), reconnect with a brand-new client,
    confirm the preference is still there."""
    filters = SearchFilters(rewritten_query="pink handbags", color="Pink")
    llm = _fake_llm(filters=filters, response_text="ok")
    collection = _fake_collection()
    encoder = MagicMock()

    with patch("src.agent.graph.search", return_value=[]):
        agent = ShoppingAgent(
            graph=build_graph(llm, collection, encoder), persistent_memory=persistent_memory
        )
        agent.chat(user_id="user-pink", session_id="s1", message="pink handbags?")

    fresh_memory = PersistentMemory(redis_client=redis.from_url(TEST_REDIS_URL, decode_responses=True))
    assert fresh_memory.load("user-pink") == UserPreferences(preferred_colors=["Pink"])


def _noop_memory():
    """A `PersistentMemory` whose `merge` is a no-op — for graph-wiring tests that don't
    care about the persistence path (keeps them independent of a running Redis)."""
    memory = MagicMock(spec=PersistentMemory)
    memory.merge.return_value = UserPreferences()
    return memory
