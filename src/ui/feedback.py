"""👍/👎 feedback persistence — SQLite, per `paths.feedback_db` (docs/ShopTalk_Plan.md Phase 7/9).

A `(user_id, session_id, query, item_id)` row is upserted on every click — re-clicking
the opposite verdict overwrites rather than piling up rows, so "the latest verdict for
this product on this query" is always a single, unambiguous record. This is exactly the
shape Phase 9's hard-negative aggregation needs: `verdict='down'` rows are candidate hard
negatives for the next triplet-mining round; `verdict='up'` rows bias personalization.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.common.config import load_config, resolve_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    item_id TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('up', 'down')),
    UNIQUE (user_id, query, item_id)
)
"""


@dataclass
class FeedbackStore:
    db_path: Path

    def record(self, *, user_id: str, session_id: str, query: str, item_id: str, verdict: str) -> None:
        if verdict not in ("up", "down"):
            raise ValueError(f"verdict must be 'up' or 'down', got {verdict!r}")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback (ts, user_id, session_id, query, item_id, verdict)
                VALUES (:ts, :user_id, :session_id, :query, :item_id, :verdict)
                ON CONFLICT (user_id, query, item_id) DO UPDATE SET
                    ts = excluded.ts, session_id = excluded.session_id, verdict = excluded.verdict
                """,
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "query": query,
                    "item_id": item_id,
                    "verdict": verdict,
                },
            )

    def verdict_for(self, *, user_id: str, query: str, item_id: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT verdict FROM feedback WHERE user_id = ? AND query = ? AND item_id = ?",
                (user_id, query, item_id),
            ).fetchone()
        return row[0] if row else None

    def all_with_verdict(self, verdict: str) -> list[dict]:
        """All rows with a given verdict — Phase 9 reads `verdict='down'` rows here as
        hard-negative candidates for the next triplet-mining round."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, user_id, session_id, query, item_id, verdict FROM feedback WHERE verdict = ? ORDER BY ts",
                (verdict,),
            ).fetchall()
        return [dict(row) for row in rows]


def load_feedback_store(db_path: Path | str | None = None) -> FeedbackStore:
    path = Path(db_path) if db_path else resolve_path(load_config()["paths"]["feedback_db"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(_SCHEMA)
    return FeedbackStore(db_path=path)
