"""Pluggable text encoder — the comparison harness behind the "compare >= 3 pretrained
encoders" requirement (`models.text_encoder` in config.yaml). Shared by the offline indexer
and the online retrieval API, so a query is embedded with *exactly* the same model, prefix
convention, and normalization it was indexed with (rubric #3 "same transformers at train
and inference").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.common.device import resolve_device
from src.common.logging import get_logger
from src.common.timer import Timer

logger = get_logger(__name__)

# model name -> (query_prefix, passage_prefix). Asymmetric retrieval encoders are TRAINED
# with these exact instruction strings baked into their contrastive-pretraining pairs —
# using the wrong prefix (or none) measurably handicaps them. This isn't a style choice;
# it's part of running a fair comparison. Symmetric encoders (MiniLM) get empty strings.
# Add an entry here before encoding with a new model.
_PREFIX_CONVENTIONS: dict[str, tuple[str, str]] = {
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
}


@dataclass
class Encoder:
    """A loaded sentence-embedding model, pinned to one device, with its prefix convention."""

    model_name: str
    model: SentenceTransformer
    device: str
    query_prefix: str
    passage_prefix: str

    def _encode(self, texts: list[str], prefix: str, *, batch_size: int = 64) -> np.ndarray:
        prefixed = [prefix + t for t in texts] if prefix else texts
        with Timer(f"encode[{self.model_name}] n={len(texts)}"):
            return self.model.encode(
                prefixed,
                batch_size=batch_size,
                normalize_embeddings=True,  # cosine similarity == dot product; matches MTEB eval
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def encode_queries(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        """Embed user queries with this model's query-side instruction prefix."""
        return self._encode(texts, self.query_prefix, batch_size=batch_size)

    def encode_passages(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        """Embed product `doc_text` with this model's passage-side instruction prefix.

        Named "passages" (not "documents") to match the asymmetric-retrieval-encoder
        convention these prefixes come from (e5's `passage: `, bge's passage-side no-op).
        """
        return self._encode(texts, self.passage_prefix, batch_size=batch_size)


def load_encoder(model_name: str, *, device: str | None = None) -> Encoder:
    """Load a registered text encoder onto `device` (auto-detected if not given).

    Raises on an unregistered model name — the explicit `_PREFIX_CONVENTIONS` registry is
    what keeps a new model from silently being encoded with the wrong (or no) instruction
    prefix, which would quietly bias the encoder comparison.
    """
    if model_name not in _PREFIX_CONVENTIONS:
        raise ValueError(
            f"Unknown text encoder {model_name!r} — register its (query_prefix, passage_prefix) "
            "convention in _PREFIX_CONVENTIONS before encoding with it."
        )
    device = device or resolve_device()
    query_prefix, passage_prefix = _PREFIX_CONVENTIONS[model_name]

    logger.info(
        "Loading text encoder %s onto %s (query_prefix=%r, passage_prefix=%r)",
        model_name,
        device,
        query_prefix,
        passage_prefix,
    )
    model = SentenceTransformer(model_name, device=device)
    return Encoder(
        model_name=model_name,
        model=model,
        device=device,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
