"""Unit tests for src.index.build — Chroma collection construction and structured-filter
search. Chroma's client/collection are faked throughout (a real PersistentClient writes to
disk and is integration-test territory); what we assert on is the part most likely to be
silently wrong: text-only vs. caption-enriched doc rebuilding, metadata sanitization
(Chroma rejects None), collection naming, and that search embeds the query (not the
passage) side and forwards the structured `where` filter untouched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.index.build import (
    build_collection,
    collection_name,
    load_collection,
    search,
)

# ---------------------------------------------------------------------------
# collection_name
# ---------------------------------------------------------------------------


def test_collection_name_normalizes_slashes_and_dots():
    assert (
        collection_name("BAAI/bge-base-en-v1.5", "caption_enriched") == "bge-base-en-v1-5__caption_enriched"
    )
    assert (
        collection_name("sentence-transformers/all-MiniLM-L6-v2", "text_only")
        == "all-MiniLM-L6-v2__text_only"
    )


# ---------------------------------------------------------------------------
# build_collection
# ---------------------------------------------------------------------------


def _products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "P1",
                "name": "Mid-Century Walnut Chair",
                "brand": "Rivet",
                "product_type": "CHAIR",
                "color": "Walnut",
                "material": "Wood",
                "bullet_points": ["Solid wood frame"],
                "keywords": ["chair"],
                "visual_caption": "a wooden chair with armrests",
            },
            {
                "item_id": "P2",
                "name": "Plain Steel Stool",
                "brand": None,
                "product_type": "STOOL",
                "color": "Silver",
                "material": None,
                "bullet_points": [],
                "keywords": ["stool"],
                "visual_caption": None,
            },
        ]
    )


def _fake_encoder(model_name="BAAI/bge-base-en-v1.5"):
    encoder = MagicMock(model_name=model_name)
    encoder.encode_passages.side_effect = lambda texts, **kw: np.ones((len(texts), 4))
    encoder.encode_queries.side_effect = lambda texts, **kw: np.ones((len(texts), 4))
    return encoder


def _fake_client(existing_names: list[str] | None = None):
    client = MagicMock()
    client.list_collections.return_value = [MagicMock(name=n) for n in (existing_names or [])]
    # MagicMock(name=...) sets the mock's repr name, not the `.name` attribute — set it explicitly.
    for mock_col, n in zip(client.list_collections.return_value, existing_names or []):
        mock_col.name = n
    collection = MagicMock()
    client.create_collection.return_value = collection
    client.get_collection.return_value = collection
    return client, collection


def test_build_collection_uses_caption_enriched_doc_text_only_for_that_corpus():
    df = _products_df()
    encoder = _fake_encoder()
    client, collection = _fake_client()

    with patch("src.index.build._client", return_value=client):
        build_collection(df, encoder, "caption_enriched", batch_size=10)

    add_kwargs = collection.add.call_args.kwargs
    assert "visual: a wooden chair with armrests" in add_kwargs["documents"][0]
    # P2 has no caption — its doc_text must NOT contain a `visual:` segment either way.
    assert "visual:" not in add_kwargs["documents"][1]


def test_build_collection_text_only_never_includes_visual_segment_even_when_caption_exists():
    df = _products_df()
    encoder = _fake_encoder()
    client, collection = _fake_client()

    with patch("src.index.build._client", return_value=client):
        build_collection(df, encoder, "text_only", batch_size=10)

    add_kwargs = collection.add.call_args.kwargs
    # P1 HAS a visual_caption, but the text_only corpus must exclude it — that's the whole
    # point of the comparison (does the image-derived signal help, in isolation).
    assert "visual:" not in add_kwargs["documents"][0]
    assert "Mid-Century Walnut Chair" in add_kwargs["documents"][0]


def test_build_collection_drops_none_valued_metadata_fields():
    df = _products_df()
    encoder = _fake_encoder()
    client, collection = _fake_client()

    with patch("src.index.build._client", return_value=client):
        build_collection(df, encoder, "text_only", batch_size=10)

    metadatas = collection.add.call_args.kwargs["metadatas"]
    # P2 has brand=None, material=None — Chroma rejects None metadata values outright.
    assert "brand" not in metadatas[1]
    assert "material" not in metadatas[1]
    assert metadatas[1] == {"product_type": "STOOL", "color": "Silver"}
    assert metadatas[0] == {"product_type": "CHAIR", "color": "Walnut", "material": "Wood", "brand": "Rivet"}


def test_build_collection_replaces_an_existing_collection_of_the_same_name():
    df = _products_df()
    encoder = _fake_encoder()
    existing_name = collection_name(encoder.model_name, "text_only")
    client, _ = _fake_client(existing_names=[existing_name])

    with patch("src.index.build._client", return_value=client):
        build_collection(df, encoder, "text_only", batch_size=10)

    client.delete_collection.assert_called_once_with(existing_name)
    client.create_collection.assert_called_once()


def test_build_collection_embeds_with_encode_passages_in_batches():
    df = _products_df()
    encoder = _fake_encoder()
    client, collection = _fake_client()

    with patch("src.index.build._client", return_value=client):
        build_collection(df, encoder, "text_only", batch_size=1)

    # Two rows, batch_size=1 -> two separate encode_passages + add calls, one row each.
    assert encoder.encode_passages.call_count == 2
    assert collection.add.call_count == 2
    encoder.encode_queries.assert_not_called()


# ---------------------------------------------------------------------------
# load_collection / search
# ---------------------------------------------------------------------------


def test_load_collection_resolves_by_model_name_and_corpus_type():
    client, collection = _fake_client()
    with patch("src.index.build._client", return_value=client):
        result = load_collection("BAAI/bge-base-en-v1.5", "caption_enriched")

    client.get_collection.assert_called_once_with("bge-base-en-v1-5__caption_enriched")
    assert result is collection


def test_search_embeds_the_query_side_and_forwards_where_filter():
    encoder = _fake_encoder()
    collection = MagicMock()
    collection.query.return_value = {"ids": [["P1", "P2"]]}

    where = {"$and": [{"color": "Blue"}, {"product_type": "CHAIR"}]}
    ids = search(collection, encoder, "a blue chair", top_k=5, where=where)

    assert ids == ["P1", "P2"]
    encoder.encode_queries.assert_called_once_with(["a blue chair"])
    encoder.encode_passages.assert_not_called()
    _, kwargs = collection.query.call_args
    assert kwargs["n_results"] == 5
    assert kwargs["where"] == where
