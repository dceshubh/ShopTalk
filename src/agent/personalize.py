"""Feedback-driven personalization (docs/ShopTalk_Plan.md Phase 9).

Re-ranks an already similarity-ranked candidate pool for one user using two signals that
are *already paid for* elsewhere in the stack — no extra model call, no extra latency
budget:

- **Direct demotion of previously-👎'd items.** A 👎 is a durable "not for me" signal that
  should hold beyond the exact query it was given on — re-surfacing the same product for a
  *similar* future search would make the feedback buttons feel like they do nothing.
- **A boost for previously-👍'd items and `preferred_colors`** (the persisted side of
  `src.agent.memory.PersistentMemory`) — "more like the ones you liked," the
  cross-session personalization the rubric names.

A user with no feedback/preference history gets back exactly `candidate_ids[:top_k]` —
personalization is a *re-ordering* of the retrieval result, never a different result set,
so the "no hallucinated products" structural guarantee (see `src.agent.graph` docstring)
still holds: every id returned still came from the real similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.memory import UserPreferences
from src.common.logging import get_logger
from src.ui.feedback import FeedbackStore

logger = get_logger(__name__)

# Score deltas, chosen so a single 👎 always outweighs any combination of boosts (a "not for
# me" signal must never be out-voted by a color match) while a re-surfaced 👍 and a
# preferred-color match are comparable, complementary nudges in the same direction.
_DOWNVOTE_PENALTY = 1_000
_LIKED_BEFORE_BOOST = 5
_PREFERRED_COLOR_BOOST = 3


@dataclass
class Personalizer:
    feedback_store: FeedbackStore

    def rerank(
        self,
        candidate_ids: list[str],
        *,
        user_id: str,
        preferences: UserPreferences,
        metadata_by_id: dict[str, dict],
        top_k: int,
    ) -> list[str]:
        """Reorder `candidate_ids` (best-first) for `user_id` and return the top `top_k`.

        The candidate pool's original rank is the base score — broken only by the signals
        below — so a user with no history gets back `candidate_ids[:top_k]` unchanged, and
        the only ids that can ever be returned are ones the real similarity search produced.
        """
        liked_ids = {
            row["item_id"] for row in self.feedback_store.all_with_verdict("up") if row["user_id"] == user_id
        }
        downvoted_ids = {
            row["item_id"]
            for row in self.feedback_store.all_with_verdict("down")
            if row["user_id"] == user_id
        }
        preferred_colors = set(preferences.preferred_colors)

        def _score(rank: int, item_id: str) -> int:
            score = len(candidate_ids) - rank
            if item_id in downvoted_ids:
                score -= _DOWNVOTE_PENALTY
            if item_id in liked_ids:
                score += _LIKED_BEFORE_BOOST
            if metadata_by_id.get(item_id, {}).get("color") in preferred_colors:
                score += _PREFERRED_COLOR_BOOST
            return score

        ranked = sorted(enumerate(candidate_ids), key=lambda pair: _score(*pair), reverse=True)
        personalized = [item_id for _, item_id in ranked][:top_k]
        if liked_ids or downvoted_ids or preferred_colors:
            logger.info(
                "Personalized ranking for user_id=%s: %d liked, %d downvoted, preferred_colors=%s -> %s",
                user_id,
                len(liked_ids),
                len(downvoted_ids),
                sorted(preferred_colors),
                personalized,
            )
        return personalized
