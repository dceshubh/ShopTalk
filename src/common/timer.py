"""Timer — a small reusable latency-measurement context manager.

Used everywhere we need per-stage timing: captioning batches, retrieval/rerank/generation
stages in the agent, and the end-to-end P95/P99 latency benchmark. Keeping one implementation
means every latency number in the project is measured the same way (an "apples to apples"
property worth calling out in the architecture doc).
"""

from __future__ import annotations

import time
from types import TracebackType

from src.common.logging import get_logger

logger = get_logger(__name__)


class Timer:
    """Context manager that records elapsed wall-clock time in milliseconds.

    Example:
        with Timer("embed_query") as t:
            vector = encoder.encode(query)
        print(t.elapsed_ms)
    """

    def __init__(self, label: str, *, log: bool = True):
        self.label = label
        self.log = log
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self.log:
            logger.info("%s took %.2f ms", self.label, self.elapsed_ms)
