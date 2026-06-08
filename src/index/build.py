"""ChromaDB index construction — one persisted collection per (encoder, corpus) cell of
the retrieval-comparison sweep (docs/ShopTalk_Plan.md §3), plus the structured-metadata
pre-filter that turns "blue chair" from "hope semantic similarity keeps red sofas out of
top-K" into "filter to color=Blue ∧ product_type=CHAIR, then rank by similarity within
that subset" — the "SQL-then-semantic" pattern (study-guide §9.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import chromadb
import pandas as pd
from chromadb.config import Settings

from src.common.config import load_config, resolve_path
from src.common.logging import get_logger
from src.common.timer import Timer
from src.embeddings.encode import Encoder
from src.preprocess.clean import build_doc_text

logger = get_logger(__name__)

CorpusType = Literal["text_only", "caption_enriched"]

# Structured side of "SQL-then-semantic" — attributes a shopper would actually filter on.
# `price` would join this list if the catalog had one (it doesn't; see PROJECT_REPORT §1).
METADATA_COLUMNS = ["product_type", "color", "material", "brand"]


def collection_name(model_name: str, corpus_type: CorpusType) -> str:
    """`BAAI/bge-base-en-v1.5` + `caption_enriched` -> `bge-base-en-v1-5__caption_enriched`.

    Chroma collection names must avoid `/`; normalizing dots too keeps every collection in
    a sweep visually parallel (`<short-model-name>__<corpus_type>`).
    """
    short_name = model_name.split("/")[-1].replace(".", "-")
    return f"{short_name}__{corpus_type}"


def _doc_text_for(row: pd.Series, corpus_type: CorpusType) -> str:
    """Rebuild the canonical doc string via the SAME `build_doc_text` the captioning
    pipeline and online API use — sharing one function is what makes "text-only vs.
    caption-enriched" an apples-to-apples toggle (just `visual_caption=None` or not)
    rather than two divergent code paths that could quietly drift.
    """
    visual_caption = row.get("visual_caption") if corpus_type == "caption_enriched" else None
    return build_doc_text(
        name=row["name"],
        brand=row["brand"],
        product_type=row["product_type"],
        color=row["color"],
        material=row["material"],
        bullet_points=list(row["bullet_points"]),
        keywords=list(row["keywords"]),
        visual_caption=visual_caption,
    )


def _metadata_for(row: pd.Series) -> dict[str, str]:
    """Chroma metadata values must be non-null str/int/float/bool — many ABO listings have
    no `material` or `brand`, so we drop missing attributes rather than store `None`.
    A `where={"color": "Blue"}` filter then simply never matches an item that lacks the
    attribute, instead of every query needing to special-case it.
    """
    return {col: row[col] for col in METADATA_COLUMNS if row.get(col) is not None}


def _client(chroma_dir: Path | str | None = None) -> chromadb.ClientAPI:
    chroma_dir = chroma_dir or resolve_path(load_config()["paths"]["chroma_dir"])
    return chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))


def build_collection(
    df: pd.DataFrame,
    encoder: Encoder,
    corpus_type: CorpusType,
    *,
    chroma_dir: Path | str | None = None,
    batch_size: int = 64,
) -> chromadb.Collection:
    """Embed every row's canonical doc string with `encoder` and persist vector + metadata
    + raw text into a fresh Chroma collection named `collection_name(encoder.model_name,
    corpus_type)`.

    Always rebuilds from scratch: indexing is a deterministic, fully-reproducible batch
    step (rerun it, get the same collection), so "stale partial index from a half-finished
    run" is a class of bug we sidestep by never doing incremental upserts here.
    """
    client = _client(chroma_dir)
    name = collection_name(encoder.model_name, corpus_type)
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    doc_texts = [_doc_text_for(row, corpus_type) for _, row in df.iterrows()]
    metadatas = [_metadata_for(row) for _, row in df.iterrows()]
    ids = df["item_id"].tolist()

    with Timer(f"build_collection[{name}] n={len(df)}"):
        for start in range(0, len(df), batch_size):
            end = start + batch_size
            embeddings = encoder.encode_passages(doc_texts[start:end], batch_size=batch_size)
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings.tolist(),
                documents=doc_texts[start:end],
                metadatas=metadatas[start:end],
            )

    logger.info("Indexed %d/%d docs into collection %r", collection.count(), len(df), name)
    return collection


def load_collection(
    model_name: str, corpus_type: CorpusType, *, chroma_dir: Path | str | None = None
) -> chromadb.Collection:
    """Reopen a previously-built collection by `(model_name, corpus_type)` — the online API
    and the eval harness both resolve collections this way so neither has to know the
    `collection_name` encoding scheme directly."""
    return _client(chroma_dir).get_collection(collection_name(model_name, corpus_type))


def search(
    collection: chromadb.Collection,
    encoder: Encoder,
    query: str,
    *,
    top_k: int = 10,
    where: dict | None = None,
) -> list[str]:
    """Embed `query` with the SAME encoder the collection was built with — the "same
    transformers train<->inference" property the rubric names explicitly — run ANN search
    (optionally structured-pre-filtered via `where`), and return ranked `item_id`s.
    """
    query_embedding = encoder.encode_queries([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
    return results["ids"][0]
