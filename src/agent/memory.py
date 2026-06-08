"""Two-tier agent memory (docs/ShopTalk_Plan.md §2.5 / Phase 5):

- **Short-term / working memory** — `ConversationBuffer`, an in-RAM per-session message
  list. Cheap, ephemeral, gone when the process restarts — exactly what "history-aware
  query rewriting within one chat" needs and no more.
- **Persistent memory** — `PersistentMemory`, a Redis-backed `UserPreferences` store keyed
  by `user_id`, surviving across sessions and process restarts. This is what lets "I
  usually shop for my 5-year-old" expressed in session 1 silently inform session 2's
  results — the personalization hook (Phase 9 / problem-doc optional deliverable).

Both are intentionally dumb stores: *what* goes into them is decided by the graph's
filter-extraction node (`src.agent.filters`), not by memory itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import redis
from pydantic import BaseModel

from src.common.config import load_config
from src.common.logging import get_logger

logger = get_logger(__name__)

_PREFS_KEY_PREFIX = "shoptalk:user_prefs:"


# ---------------------------------------------------------------------------
# Short-term memory — in-RAM conversation buffer
# ---------------------------------------------------------------------------


@dataclass
class ConversationBuffer:
    """Per-session message history, capped at `max_turns` user/assistant pairs.

    Capping (not unbounded growth) matters for two reasons: it keeps the prompt within the
    generator's context window as a chat goes long, and it bounds per-request latency —
    both real "production-realistic" concerns, not just tidiness.
    """

    max_turns: int = 10
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self._append("user", content)

    def add_assistant(self, content: str) -> None:
        self._append("assistant", content)

    def _append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        max_messages = self.max_turns * 2  # one user + one assistant message per turn
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def as_messages(self) -> list[dict[str, str]]:
        """Return the buffered turns as chat-message dicts, ready to prepend to a new
        LLM call (after the system prompt, before the current user turn)."""
        return list(self.messages)


# ---------------------------------------------------------------------------
# Persistent memory — Redis-backed user preferences
# ---------------------------------------------------------------------------


class UserPreferences(BaseModel):
    """Structured, cross-session shopper profile. Every field is optional — a brand-new
    user has an empty profile, and fields fill in incrementally as the filter-extraction
    node notices them in conversation (e.g. "for my 5-year-old" -> `typical_recipient`)."""

    typical_recipient: str | None = None
    budget_ceiling: float | None = None
    preferred_size: str | None = None
    preferred_colors: list[str] = []


@dataclass
class PersistentMemory:
    """Redis-backed `UserPreferences` store, one JSON blob per `user_id`."""

    redis_client: redis.Redis

    def load(self, user_id: str) -> UserPreferences:
        """Read the stored profile, or an empty one for a never-seen user_id — the
        "no prefs yet" case is not an error, it's the default."""
        raw = self.redis_client.get(_PREFS_KEY_PREFIX + user_id)
        if raw is None:
            return UserPreferences()
        return UserPreferences.model_validate_json(raw)

    def save(self, user_id: str, prefs: UserPreferences) -> None:
        self.redis_client.set(_PREFS_KEY_PREFIX + user_id, prefs.model_dump_json())

    def merge(self, user_id: str, partial: UserPreferences) -> UserPreferences:
        """Layer newly-extracted fields onto the existing profile and persist the result.

        Only non-empty fields in `partial` overwrite the stored profile — a turn that
        doesn't mention budget must not erase a budget learned three turns (or three
        sessions) ago. Returns the merged profile so the caller can use it immediately
        without a round-trip re-read.
        """
        current = self.load(user_id)
        updates = {
            field_name: value
            for field_name, value in partial.model_dump().items()
            if value not in (None, [], "")
        }
        merged = current.model_copy(update=updates)
        self.save(user_id, merged)
        return merged


def load_persistent_memory(redis_url: str | None = None) -> PersistentMemory:
    """Connect to Redis using `agent.memory.redis_url` from config.yaml unless overridden.
    `decode_responses=True` so `.get()` returns `str` (matching `model_validate_json`'s
    expected input) rather than raw `bytes`."""
    redis_url = redis_url or load_config()["agent"]["memory"]["redis_url"]
    logger.info("Connecting persistent agent memory to Redis at %s", redis_url)
    return PersistentMemory(redis_client=redis.from_url(redis_url, decode_responses=True))
