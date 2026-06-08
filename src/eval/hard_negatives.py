"""Hard-negative mining from 👍/👎 feedback — closes the loop back to fine-tuning
(docs/ShopTalk_Plan.md Phase 9 -> Phase 4).

Both of `finetune.loss_primary` (`MultipleNegativesRankingLoss`) and `.loss_alt`
(`TripletLoss`) train on `(anchor, positive, negative)` triples. Without explicit hard
negatives, `MultipleNegativesRankingLoss` falls back to random in-batch negatives — usually
nothing like what a real shopper would actually confuse with the right answer. A 👎 is
the opposite: **direct, human-labeled evidence that "for this exact query, this product was
NOT the right answer"** — about as hard a negative as a dataset can contain, for free, as a
byproduct of normal use.

**Pairing rule:** a 👎 only becomes a triplet when paired with a 👍 from the SAME user on
the SAME query — the query is then a true `(anchor, positive, negative)` triple where all
three legs are about the same search intent (a 👎 paired with an unrelated 👍 from a
different query would teach the encoder a false association, which is worse than no signal
at all). A 👎 with no matching same-query 👍 is skipped, not force-paired — it stays in the
raw feedback table for the next mining round once a positive appears.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.ui.feedback import FeedbackStore


@dataclass(frozen=True)
class HardNegativeTriplet:
    """One `(anchor, positive, negative)` triple, ready for `MultipleNegativesRankingLoss`
    / `TripletLoss` once `item_id`s are resolved to their canonical doc text (the same
    `build_doc_text` the index and captioning pipeline already share)."""

    query: str
    positive_item_id: str
    negative_item_id: str


def mine_hard_negatives(feedback_store: FeedbackStore) -> list[HardNegativeTriplet]:
    """Cross-join same-user, same-query 👍/👎 pairs into hard-negative training triplets.

    One query can yield several triplets (e.g. two 👍s and one 👎 -> two triples sharing
    the same anchor and negative) — each is a distinct, valid training signal, not
    redundant data, so all combinations are kept.
    """
    downvotes = feedback_store.all_with_verdict("down")
    upvotes = feedback_store.all_with_verdict("up")

    upvotes_by_user_query: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in upvotes:
        upvotes_by_user_query[(row["user_id"], row["query"])].append(row)

    triplets = []
    for downvote in downvotes:
        matching_upvotes = upvotes_by_user_query.get((downvote["user_id"], downvote["query"]), [])
        for upvote in matching_upvotes:
            triplets.append(
                HardNegativeTriplet(
                    query=downvote["query"],
                    positive_item_id=upvote["item_id"],
                    negative_item_id=downvote["item_id"],
                )
            )
    return triplets
