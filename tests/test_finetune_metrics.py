"""Unit tests for src.embeddings.finetune.metrics — the fine-tuned-vs-pretrained
comparison numbers for Phase 4 (docs/ShopTalk_Plan.md §2.3): separation margin and
category clustering.

The encoder is faked throughout (a real SentenceTransformer has no place in a fast unit
suite — see tests/test_encode.py for the same convention). What's under test is the
arithmetic: given known embeddings, do these functions compute the right cosine
similarities and aggregate them correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.embeddings.encode import Encoder
from src.embeddings.finetune.metrics import category_clustering, doc_text, separation_margin
from src.eval.hard_negatives import HardNegativeTriplet


def _products_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "POS1",
                "name": "Walnut Coffee Table",
                "brand": "Acme",
                "product_type": "TABLE",
                "color": "Walnut",
                "material": "Wood",
                "bullet_points": [],
                "keywords": [],
            },
            {
                "item_id": "NEG1",
                "name": "Teal Side Table",
                "brand": "Acme",
                "product_type": "TABLE",
                "color": "Teal",
                "material": "Wood",
                "bullet_points": [],
                "keywords": [],
            },
            {
                "item_id": "OTHER1",
                "name": "Black Office Chair",
                "brand": "Acme",
                "product_type": "CHAIR",
                "color": "Black",
                "material": None,
                "bullet_points": [],
                "keywords": [],
            },
        ]
    )


class _FakeModel:
    """Maps each input string to a fixed 2D vector via a lookup table, normalized like a
    real SentenceTransformer with `normalize_embeddings=True`."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = vectors

    def encode(self, texts, **_kwargs):
        out = np.array([self._vectors[t] for t in texts], dtype=float)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / norms


def _encoder(vectors: dict[str, np.ndarray]) -> Encoder:
    return Encoder(
        model_name="fake/model",
        model=_FakeModel(vectors),
        device="cpu",
        query_prefix="",
        passage_prefix="",
    )


# ---------------------------------------------------------------------------
# doc_text
# ---------------------------------------------------------------------------


def test_doc_text_matches_build_doc_text_for_a_row():
    row = _products_df().iloc[0]

    text = doc_text(row)

    assert "Walnut Coffee Table" in text
    assert "color: Walnut" in text
    assert "material: Wood" in text


# ---------------------------------------------------------------------------
# separation_margin
# ---------------------------------------------------------------------------


def test_separation_margin_is_zero_for_no_triplets():
    encoder = _encoder({})
    assert separation_margin(encoder, [], _products_df()) == 0.0


def test_separation_margin_computes_mean_positive_minus_negative_cosine():
    df = _products_df()
    query = "walnut table"
    positive_text = doc_text(df.set_index("item_id").loc["POS1"])
    negative_text = doc_text(df.set_index("item_id").loc["NEG1"])

    # query aligned with positive (cos=1), orthogonal to negative (cos=0) -> margin = 1.0
    vectors = {
        query: np.array([1.0, 0.0]),
        positive_text: np.array([1.0, 0.0]),
        negative_text: np.array([0.0, 1.0]),
    }
    encoder = _encoder(vectors)
    triplets = [HardNegativeTriplet(query=query, positive_item_id="POS1", negative_item_id="NEG1")]

    margin = separation_margin(encoder, triplets, df)

    assert margin == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# category_clustering
# ---------------------------------------------------------------------------


def test_category_clustering_separates_intra_and_inter_category_means():
    df = _products_df()  # POS1/NEG1 = TABLE, OTHER1 = CHAIR
    pos_text = doc_text(df.set_index("item_id").loc["POS1"])
    neg_text = doc_text(df.set_index("item_id").loc["NEG1"])
    other_text = doc_text(df.set_index("item_id").loc["OTHER1"])

    # Two TABLEs identical (cos=1, intra); CHAIR orthogonal to both (cos=0, inter).
    vectors = {
        pos_text: np.array([1.0, 0.0]),
        neg_text: np.array([1.0, 0.0]),
        other_text: np.array([0.0, 1.0]),
    }
    encoder = _encoder(vectors)

    result = category_clustering(encoder, df, sample_size=3, seed=42)

    assert result["intra_category_mean_cosine"] == pytest.approx(1.0)
    assert result["inter_category_mean_cosine"] == pytest.approx(0.0)


def test_category_clustering_handles_a_sample_with_only_one_category():
    df = _products_df().iloc[:2]  # both TABLE -> no inter-category pairs
    pos_text = doc_text(df.set_index("item_id").loc["POS1"])
    neg_text = doc_text(df.set_index("item_id").loc["NEG1"])

    vectors = {
        pos_text: np.array([1.0, 0.0]),
        neg_text: np.array([0.0, 1.0]),
    }
    encoder = _encoder(vectors)

    result = category_clustering(encoder, df, sample_size=2, seed=42)

    assert result["inter_category_mean_cosine"] == 0.0
