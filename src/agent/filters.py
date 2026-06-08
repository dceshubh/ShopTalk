"""Query understanding — one structured-extraction LLM call that does double duty:

1. **History-aware query rewriting**: resolves anaphora against the conversation buffer
   ("show me cheaper ones" -> "wooden dining chairs under $100", reusing "wooden dining
   chairs" from two turns back) into a standalone search string.
2. **Structured filter extraction** (Pydantic, not regex — per the project's structured-
   output convention): pulls `product_type`/`color`/`material`/`brand` into a `SearchFilters`
   object that drives Chroma's "SQL-then-semantic" pre-filter (`src.index.build.search`'s
   `where=`) — the mechanism behind "a blue-chair query never returns a red item."

One call instead of two: the rewrite and the filters both need the same context (the
conversation history + the current message), and asking the model to produce them together
is both cheaper (one round-trip) and more consistent (the filters describe the SAME
resolved intent the rewritten query expresses) than extracting them independently.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.llm import GeneratorLLM
from src.index.build import METADATA_COLUMNS

_SYSTEM_PROMPT = """You are the query-understanding stage of a shopping assistant.

Given the conversation so far and the shopper's latest message, produce:
- `rewritten_query`: a standalone search string that resolves any references to earlier
  turns (e.g. if the shopper previously asked about "wooden dining chairs" and now says
  "show me cheaper ones", the rewritten query should be something like "cheap wooden dining
  chairs", not just "cheaper ones").
- Structured filters (`product_type`, `color`, `material`, `brand`): set a field ONLY when
  the shopper's intent clearly names that attribute (their own words or a clear synonym).
  Leave a field null if it isn't clearly specified — guessing a filter the shopper didn't
  ask for would silently hide relevant results from them.
"""


class SearchFilters(BaseModel):
    """Structured query-understanding output. `rewritten_query` always has a value (it's
    the fallback search string even when no structured filter applies); the metadata
    fields are optional — most queries name only some, or none, of them."""

    rewritten_query: str = Field(description="Standalone, history-resolved search query")
    product_type: str | None = Field(default=None, description="e.g. CHAIR, RUG, SHOES")
    color: str | None = Field(default=None, description="e.g. Blue, Brown, Black")
    material: str | None = Field(default=None, description="e.g. Wood, Leather, Steel")
    brand: str | None = Field(default=None, description="A specific brand name, if named")


def extract_filters(
    llm: GeneratorLLM,
    conversation: list[dict[str, str]],
    current_message: str,
) -> SearchFilters:
    """Run the combined rewrite + filter-extraction call.

    `conversation` is the buffered history (`ConversationBuffer.as_messages()`) — passed
    as-is so the model sees the same turns the user experienced, in order.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *conversation,
        {"role": "user", "content": current_message},
    ]
    return llm.complete_structured(messages, SearchFilters)


def filters_to_where(filters: SearchFilters) -> dict | None:
    """Convert extracted filters into a Chroma `where` clause for `src.index.build.search`.

    Only `METADATA_COLUMNS` fields participate (matches what's actually stored on each
    collection — see `src.index.build._metadata_for`); `rewritten_query` is the search
    *text*, not a filter. Returns `None` when nothing was extracted — searching with
    `where=None` runs unfiltered ANN, the correct behavior for an unconstrained query like
    "show me something nice for my kitchen."
    """
    clauses = [
        {column: value}
        for column in METADATA_COLUMNS
        if (value := getattr(filters, column, None)) is not None
    ]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
