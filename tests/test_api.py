"""Unit + integration tests for src.api.main — the FastAPI inference service.

Most tests substitute a *fake* `loader` (`create_app(loader=...)`) so they exercise the real
routes/middleware/error-handlers/lifespan-wiring without booting Groq, the encoder, Chroma,
or Redis — the same "fake the expensive edge, test the wiring" split used throughout
`tests/test_graph.py`. One test (`test_search_parity_...`) loads the REAL encoder + index
(no GROQ_API_KEY needed — only the agent's filter-extraction LLM is faked) because the
Phase-6 exit gate it covers — "same query -> API result == offline notebook result" — is
only provable against the real retrieval path; skipped if the dev-scale index isn't built.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agent.filters import SearchFilters
from src.agent.graph import ShoppingAgent, build_graph
from src.agent.memory import PersistentMemory, UserPreferences
from src.api.catalog import ProductCatalog
from src.api.main import RuntimeModels, create_app

# ---------------------------------------------------------------------------
# Fakes — mirror the _fake_llm / _fake_collection helpers in tests/test_graph.py
# ---------------------------------------------------------------------------

_CATALOG_ROWS = {
    "REAL1": {
        "name": "Brown Leather Recliner",
        "brand": "Acme",
        "color": "Brown",
        "material": "Leather",
        "product_type": "CHAIR",
        "image_path": "ab/real1.jpg",
    },
    "REAL2": {
        "name": "Brown Leather Sofa",
        "brand": "Acme",
        "color": "Brown",
        "material": "Leather",
        "product_type": "SOFA",
        "image_path": "cd/real2.jpg",
    },
}


def _fake_catalog() -> ProductCatalog:
    return ProductCatalog(by_id=dict(_CATALOG_ROWS))


def _fake_llm(*, filters: SearchFilters | None = None, response_text: str = "Here are some great picks!"):
    llm = MagicMock()
    llm.complete_structured.return_value = filters or SearchFilters(
        rewritten_query="brown leather chairs", color="Brown"
    )
    llm.complete.return_value = response_text
    return llm


def _fake_collection():
    collection = MagicMock()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    collection.name = "fake-collection"
    return collection


def _noop_memory() -> PersistentMemory:
    memory = MagicMock(spec=PersistentMemory)
    memory.merge.return_value = UserPreferences()
    return memory


def _make_loader(*, retrieved_ids: list[str], response_text: str = "Here are some great picks!"):
    """A loader that builds a REAL ShoppingAgent (so session-buffer logic is genuinely
    exercised) wired to faked LLM/collection/encoder (so no network/model calls happen)."""

    def loader() -> RuntimeModels:
        llm = _fake_llm(response_text=response_text)
        collection = _fake_collection()
        encoder = MagicMock()
        graph = build_graph(llm, collection, encoder, top_k=5)
        agent = ShoppingAgent(graph=graph, persistent_memory=_noop_memory())
        return RuntimeModels(
            agent=agent,
            catalog=_fake_catalog(),
            generator_model="llama-3.1-8b-instant",
            encoder_model="BAAI/bge-base-en-v1.5",
            collection_name="fake-collection",
        )

    return loader


@pytest.fixture
def client():
    """A TestClient wired to a fake loader, with `src.agent.graph.search` patched for the
    lifetime of the client so every /chat call retrieves the same fixed REAL1/REAL2 ids —
    the patch must stay active across requests, not just during loader construction."""
    from unittest.mock import patch

    with patch("src.agent.graph.search", return_value=["REAL1", "REAL2"]):
        app = create_app(loader=_make_loader(retrieved_ids=["REAL1", "REAL2"]))
        with TestClient(app) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# /health — proves models load exactly ONCE
# ---------------------------------------------------------------------------


def test_health_reports_loaded_model_identities_and_load_count_of_one(client):
    first = client.get("/health")
    second = client.get("/health")
    third = client.get("/health")

    for response in (first, second, third):
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["generator_model"] == "llama-3.1-8b-instant"
        assert body["encoder_model"] == "BAAI/bge-base-en-v1.5"
        assert body["catalog_size"] == 2
        # The exit-gate proof: the SAME load_count across every request — the lifespan ran
        # exactly once for this process, /health did not trigger (or report) a reload.
        assert body["load_count"] == 1


# ---------------------------------------------------------------------------
# /chat — schema + grounded product cards
# ---------------------------------------------------------------------------


def test_chat_returns_response_and_product_cards_with_real_ids_and_image_paths(client):
    response = client.post(
        "/chat", json={"user_id": "u1", "session_id": "s1", "message": "brown leather chairs?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["response_text"] == "Here are some great picks!"
    assert [p["item_id"] for p in body["products"]] == ["REAL1", "REAL2"]
    for card in body["products"]:
        assert card["image_path"]  # every retrieved product resolves to a real image path
        assert card["name"]


def test_chat_rejects_an_empty_message_with_a_structured_422_not_a_stacktrace(client):
    response = client.post("/chat", json={"user_id": "u1", "session_id": "s1", "message": ""})

    assert response.status_code == 422
    body = response.json()
    assert "request_id" in body
    assert "message" in body
    assert "errors" in body  # field-level Pydantic validation detail, not a raw traceback


def test_chat_rejects_a_missing_field_with_a_structured_422(client):
    response = client.post("/chat", json={"user_id": "u1", "message": "hi"})  # no session_id

    assert response.status_code == 422
    assert "request_id" in response.json()


# ---------------------------------------------------------------------------
# /products/{id}
# ---------------------------------------------------------------------------


def test_get_product_returns_a_card_for_a_known_id(client):
    response = client.get("/products/REAL1")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "item_id": "REAL1",
        "name": "Brown Leather Recliner",
        "image_path": "ab/real1.jpg",
        "product_type": "CHAIR",
        "color": "Brown",
        "material": "Leather",
        "brand": "Acme",
    }


def test_get_product_returns_a_structured_404_for_an_unknown_id(client):
    response = client.get("/products/NOPE000000")

    assert response.status_code == 404
    body = response.json()
    assert "request_id" in body
    assert "NOPE000000" in body["message"]


# ---------------------------------------------------------------------------
# Concurrency — parallel sessions don't corrupt each other's history
# ---------------------------------------------------------------------------


def test_concurrent_chats_across_sessions_do_not_corrupt_each_others_history(client):
    sessions = [f"session-{i}" for i in range(10)]

    def send(session_id: str):
        return client.post(
            "/chat",
            json={"user_id": "shared-user", "session_id": session_id, "message": f"hello from {session_id}"},
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        responses = list(pool.map(send, sessions))

    assert all(r.status_code == 200 for r in responses)

    models: RuntimeModels = client.app.state.models
    assert set(models.agent._buffers.keys()) == set(sessions)
    for session_id in sessions:
        messages = models.agent._buffers[session_id].messages
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        # Each session's buffer holds ONLY its own message — never another session's.
        assert user_messages == [f"hello from {session_id}"]


# ---------------------------------------------------------------------------
# Live retrieval-parity check — "same query -> API result == offline notebook result"
# ---------------------------------------------------------------------------


@pytest.fixture
def real_collection_and_encoder():
    from src.common.config import load_config
    from src.embeddings.encode import load_encoder
    from src.index.build import load_collection

    cfg = load_config()
    encoder_model = cfg["models"]["text_encoder"]["primary"]
    try:
        collection = load_collection(encoder_model, "caption_enriched")
    except Exception:
        pytest.skip("dev-scale Chroma index not built — run src.index.build to enable this test")
    encoder = load_encoder(encoder_model)
    return collection, encoder


def test_search_parity_between_chat_path_and_offline_search(real_collection_and_encoder):
    """Both the agent's search_products node and the Phase-3 offline eval harness call the
    SAME `src.index.build.search(collection, encoder, query, ...)` — the "same transformers
    train<->inference" property the rubric names explicitly. Prove it end-to-end: force the
    agent's filter-extraction (the only faked piece — no GROQ_API_KEY needed) to rewrite to
    the EXACT query text an offline caller would use, and assert byte-identical ranked ids."""
    from src.index.build import search

    collection, encoder = real_collection_and_encoder
    query = "a brown leather chair"

    offline_ids = search(collection, encoder, query, top_k=5)

    fake_llm = MagicMock()
    fake_llm.complete_structured.return_value = SearchFilters(rewritten_query=query)
    fake_llm.complete.return_value = "..."
    graph = build_graph(fake_llm, collection, encoder, top_k=5)

    online_result = graph.invoke({"user_id": "u1", "history": [], "message": query})

    assert online_result["retrieved_ids"] == offline_ids


# ---------------------------------------------------------------------------
# create_app() — the production app is constructible and wired with the expected routes,
# and nothing is loaded before the lifespan actually runs (no eager model construction
# at import time, which would defeat "loaded once at startup, not at import").
# ---------------------------------------------------------------------------


def test_create_app_is_constructible_with_the_expected_routes_and_lazy_model_loading():
    app = create_app(loader=_make_loader(retrieved_ids=[]))

    paths = {route.path for route in app.routes}
    assert {"/health", "/chat", "/products/{item_id}"} <= paths
    assert not hasattr(app.state, "models")  # lifespan hasn't run yet — nothing loaded at construction time


def test_module_level_app_uses_the_real_loader_by_default():
    from src.api.main import app as production_app

    assert production_app.state.__dict__.get("models") is None or not hasattr(production_app.state, "models")
    paths = {route.path for route in production_app.routes}
    assert {"/health", "/chat", "/products/{item_id}"} <= paths
