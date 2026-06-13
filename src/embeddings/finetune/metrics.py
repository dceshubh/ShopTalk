"""Fine-tuned-vs-pretrained comparison metrics for Phase 4 (docs/ShopTalk_Plan.md §2.3).

Two numbers, reported SEPARATELY — do not conflate them:

  - `separation_margin`: mean(cos(query, positive_doc) - cos(query, negative_doc)) across
    training triplets. This is the quantity the fine-tune is DIRECTLY trained to widen —
    the primary "did fine-tuning work" number, expected to increase post-fine-tune.

  - `category_clustering`: mean intra-`product_type` cosine vs mean inter-`product_type`
    cosine over a sample of products. Reported AS-IS, with NO claim that fine-tuning
    improves it — the attribute-hard-negative scheme that widens the separation margin
    above can simultaneously *reduce* intra-category compactness (it's training the model
    to pull apart same-category-different-attribute pairs). Report both; don't promise the
    second one moves in any particular direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.embeddings.encode import Encoder
from src.eval.hard_negatives import HardNegativeTriplet
from src.preprocess.clean import build_doc_text


def doc_text(row: pd.Series) -> str:
    """Rebuild the canonical doc string for a product row via the same `build_doc_text`
    the index and captioning pipeline use (see `src.index.build._doc_text_for`)."""
    return build_doc_text(
        name=row["name"],
        brand=row["brand"],
        product_type=row["product_type"],
        color=row["color"],
        material=row["material"],
        bullet_points=list(row["bullet_points"]),
        keywords=list(row["keywords"]),
        visual_caption=row.get("visual_caption"),
    )


def separation_margin(encoder: Encoder, triplets: list[HardNegativeTriplet], df: pd.DataFrame) -> float:
    """Mean(cos(query, positive_doc) - cos(query, negative_doc)).

    Embeddings from `Encoder` are L2-normalized (see `Encoder._encode`), so cosine
    similarity reduces to a dot product.
    """
    if not triplets:
        return 0.0
    by_id = df.set_index("item_id")
    queries = [t.query for t in triplets]
    positive_texts = [doc_text(by_id.loc[t.positive_item_id]) for t in triplets]
    negative_texts = [doc_text(by_id.loc[t.negative_item_id]) for t in triplets]

    query_emb = encoder.encode_queries(queries)
    positive_emb = encoder.encode_passages(positive_texts)
    negative_emb = encoder.encode_passages(negative_texts)

    positive_sim = np.sum(query_emb * positive_emb, axis=1)
    negative_sim = np.sum(query_emb * negative_emb, axis=1)
    return float(np.mean(positive_sim - negative_sim))


def category_clustering(
    encoder: Encoder, df: pd.DataFrame, *, sample_size: int = 200, seed: int = 42
) -> dict[str, float]:
    """Mean pairwise cosine similarity within the same `product_type` vs across different
    `product_type`s, over a random sample of `sample_size` products (or all of `df` if
    smaller)."""
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    doc_texts = [doc_text(row) for _, row in sample_df.iterrows()]
    embeddings = encoder.encode_passages(doc_texts)
    product_types = sample_df["product_type"].tolist()

    sims = embeddings @ embeddings.T
    n = len(sample_df)
    intra: list[float] = []
    inter: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if product_types[i] == product_types[j] else inter).append(sims[i, j])

    return {
        "intra_category_mean_cosine": float(np.mean(intra)) if intra else 0.0,
        "inter_category_mean_cosine": float(np.mean(inter)) if inter else 0.0,
    }
