"""The LangGraph conversational-shopping graph (docs/ShopTalk_Plan.md §2.5 / Phase 5):

    understand_query -> search_products -> generate_response

- **understand_query**: history-aware query rewrite + Pydantic filter extraction
  (`src.agent.filters.extract_filters`) in one LLM call.
- **search_products**: runs the rewritten query through Chroma with the extracted
  structured pre-filter (`src.index.build.search` — "SQL-then-semantic").
- **generate_response**: phrases the retrieved products conversationally and asks an
  upsell follow-up.

**Anti-hallucination is structural, not a prompt hope**: `AgentTurn.product_ids` is always
*exactly* `retrieved_ids` from the search step — never parsed out of the LLM's free text.
The exit gate "generated answers cite real retrieved product_ids, no hallucinated products"
holds *by construction*: the product references shown to the user are the retrieval
result, full stop; the LLM only ever narrates them, it never gets to invent which products
are "shown."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.filters import SearchFilters, extract_filters, filters_to_where
from src.agent.llm import GeneratorLLM
from src.agent.memory import ConversationBuffer, PersistentMemory, UserPreferences
from src.agent.personalize import Personalizer
from src.common.logging import get_logger
from src.index.build import search

logger = get_logger(__name__)

_GENERATION_SYSTEM_PROMPT = """You are ShopTalk, a friendly shopping assistant.

You will be given a shopper's request and a list of retrieved products (name, type, color,
material, brand — whatever is known). Using ONLY the products listed:
- Recommend the most relevant one(s) and briefly say why they fit the request.
- If nothing in the list fits well, say so honestly rather than forcing a recommendation.
- End with a short, natural upsell follow-up question (e.g. "Would you like to see similar
  options in a different color?" or "Want me to check matching items too?").

Do not mention products that are not in the provided list — you have no knowledge of the
catalog beyond what's given to you for this turn.
"""


class AgentState(TypedDict, total=False):
    user_id: str
    history: list[dict[str, str]]
    message: str
    filters: SearchFilters
    retrieved_ids: list[str]
    response_text: str


@dataclass
class AgentTurn:
    """One assistant turn. `product_ids` is sourced ENTIRELY from retrieval — see module
    docstring on why this is what makes "no hallucinated products" true by construction."""

    response_text: str
    product_ids: list[str]
    filters: SearchFilters


# ---------------------------------------------------------------------------
# Graph nodes — each takes/returns a partial AgentState (LangGraph merges the dicts)
# ---------------------------------------------------------------------------


def _understand_query_node(llm: GeneratorLLM):
    def node(state: AgentState) -> dict:
        filters = extract_filters(llm, state["history"], state["message"])
        logger.info("Extracted filters: %s", filters)
        return {"filters": filters}

    return node


def _metadata_for_ids(collection, item_ids: list[str]) -> dict[str, dict]:
    if not item_ids:
        return {}
    got = collection.get(ids=item_ids, include=["metadatas"])
    return dict(zip(got["ids"], got["metadatas"]))


def _search_products_node(
    collection,
    encoder,
    personalizer: Personalizer,
    persistent_memory: PersistentMemory,
    *,
    top_k: int,
    pool_size: int,
):
    """Retrieve a `pool_size`-deep similarity-ranked candidate pool (deeper than the
    `top_k` actually shown), then let `Personalizer.rerank` reorder it for this user before
    truncating to `top_k`. Pulling a deeper pool is what gives personalization something to
    *work with* — re-ranking only the top 10 of an already-10-deep search couldn't surface
    a previously-liked item ranked 11th by raw similarity."""

    def node(state: AgentState) -> dict:
        filters = state["filters"]
        where = filters_to_where(filters)
        candidate_ids = search(collection, encoder, filters.rewritten_query, top_k=pool_size, where=where)
        preferences = persistent_memory.load(state["user_id"])
        metadata_by_id = _metadata_for_ids(collection, candidate_ids)
        retrieved_ids = personalizer.rerank(
            candidate_ids,
            user_id=state["user_id"],
            preferences=preferences,
            metadata_by_id=metadata_by_id,
            top_k=top_k,
        )
        logger.info(
            "Retrieved %d candidates, personalized to %d products for query %r (where=%s)",
            len(candidate_ids),
            len(retrieved_ids),
            filters.rewritten_query,
            where,
        )
        return {"retrieved_ids": retrieved_ids}

    return node


def _generate_response_node(llm: GeneratorLLM, collection):
    def node(state: AgentState) -> dict:
        retrieved_ids = state["retrieved_ids"]
        product_lines = _describe_products(collection, retrieved_ids)
        grounding = (
            "Retrieved products:\n" + "\n".join(product_lines)
            if product_lines
            else "Retrieved products: (none matched this search)"
        )
        messages = [
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            *state["history"],
            {"role": "user", "content": f"{state['message']}\n\n{grounding}"},
        ]
        response_text = llm.complete(messages)
        return {"response_text": response_text}

    return node


def _describe_products(collection, item_ids: list[str]) -> list[str]:
    """Render each retrieved item's known attributes as one grounding line for the
    generator — the only catalog information the LLM is allowed to draw on this turn."""
    if not item_ids:
        return []
    got = collection.get(ids=item_ids, include=["documents", "metadatas"])
    lines = []
    for item_id, document, metadata in zip(got["ids"], got["documents"], got["metadatas"]):
        attrs = ", ".join(f"{k}={v}" for k, v in metadata.items())
        lines.append(f"- [{item_id}] {document[:200]} ({attrs})")
    return lines


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    llm: GeneratorLLM,
    collection,
    encoder,
    personalizer: Personalizer,
    persistent_memory: PersistentMemory,
    *,
    top_k: int = 10,
    pool_size: int = 30,
):
    """Wire the three nodes into a compiled LangGraph: understand -> search -> generate.

    `pool_size` (default 30, three times `top_k`) is the personalization headroom — see
    `_search_products_node` docstring for why re-ranking needs a deeper pool than the final
    shown count to have anything to work with.
    """
    graph = StateGraph(AgentState)
    graph.add_node("understand_query", _understand_query_node(llm))
    graph.add_node(
        "search_products",
        _search_products_node(
            collection, encoder, personalizer, persistent_memory, top_k=top_k, pool_size=pool_size
        ),
    )
    graph.add_node("generate_response", _generate_response_node(llm, collection))

    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "search_products")
    graph.add_edge("search_products", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Orchestration — owns per-session short-term memory and cross-session persistent memory
# ---------------------------------------------------------------------------


@dataclass
class ShoppingAgent:
    """The thing `src.api` talks to: one compiled graph + a conversation buffer per
    `session_id` (short-term, in-RAM — gone on restart, by design) + one shared
    `PersistentMemory` (Redis-backed — survives restarts, by design)."""

    graph: object
    persistent_memory: PersistentMemory
    max_turns: int = 10
    _buffers: dict[str, ConversationBuffer] = field(default_factory=dict)

    def chat(self, *, user_id: str, session_id: str, message: str) -> AgentTurn:
        buffer = self._buffers.setdefault(session_id, ConversationBuffer(max_turns=self.max_turns))

        result = self.graph.invoke({"user_id": user_id, "history": buffer.as_messages(), "message": message})
        filters: SearchFilters = result["filters"]
        turn = AgentTurn(
            response_text=result["response_text"],
            product_ids=list(result["retrieved_ids"]),  # structural anti-hallucination — see module docstring
            filters=filters,
        )

        buffer.add_user(message)
        buffer.add_assistant(turn.response_text)
        self.persistent_memory.merge(user_id, _prefs_from_filters(filters))
        return turn


def _prefs_from_filters(filters: SearchFilters) -> UserPreferences:
    """Project this turn's extracted filters onto the persistent-preference schema —
    only `color` maps directly today (`preferred_colors`); other `UserPreferences` fields
    (`typical_recipient`, `budget_ceiling`, `preferred_size`) need richer extraction than
    today's catalog-attribute filters carry, and are left for `PersistentMemory.merge` to
    leave untouched (it never erases a field the current turn didn't mention)."""
    return UserPreferences(preferred_colors=[filters.color] if filters.color else [])
