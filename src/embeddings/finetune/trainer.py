"""LoRA fine-tuning of the text encoder (docs/ShopTalk_Plan.md Phase 4, §2.3).

Reproducible from a single command + config (`finetune` section of `configs/config.yaml`):

    python -m src.embeddings.finetune.trainer

Loads `finetune.base_model` (default `BAAI/bge-base-en-v1.5`), wraps its transformer
backbone's attention projections with a LoRA adapter (`peft`), trains with
`MultipleNegativesRankingLoss` on the triplets from
`src.embeddings.finetune.triplet_mining.build_training_triplets`, and saves the adapted
model to a versioned path under `weights/finetuned/` (not pushed to git — see
`.gitignore`/`.dockerignore`).

Run on a GPU (Kaggle/Colab free T4 — see `notebooks/03_finetune_bge_lora_kaggle.ipynb`);
the dev-scale dataset trains in well under an hour per docs/ShopTalk_Plan.md §8.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from peft import LoraConfig, get_peft_model
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.common.config import load_config, resolve_path
from src.common.logging import get_logger
from src.embeddings.finetune.metrics import doc_text
from src.embeddings.finetune.triplet_mining import build_training_triplets
from src.eval.hard_negatives import HardNegativeTriplet
from src.eval.harness import load_golden_set

logger = get_logger(__name__)

# BERT-family self-attention projections — the modules `bge-base-en-v1.5` (and e5/MiniLM,
# all BERT-derived) expose under these names. Add an entry here (or generalize) before
# LoRA-tuning a non-BERT-family base model.
LORA_TARGET_MODULES = ["query", "key", "value"]


def apply_lora(model: SentenceTransformer, lora_cfg: dict) -> SentenceTransformer:
    """Wrap `model`'s transformer backbone with a LoRA adapter, in place, per
    `finetune.lora` (`r`, `alpha`, `dropout`) in `configs/config.yaml`."""
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
    return model


def triplets_to_examples(triplets: list[HardNegativeTriplet], df: pd.DataFrame) -> list[InputExample]:
    """`(query, positive_doc, negative_doc)` -> `InputExample` — the shape
    `MultipleNegativesRankingLoss`/`TripletLoss` both consume. Triplets referencing an
    `item_id` not present in `df` are dropped (defensive — `build_training_triplets`
    shouldn't produce these, but a stale/partial `df` could)."""
    by_id = df.set_index("item_id")
    return [
        InputExample(
            texts=[
                triplet.query,
                doc_text(by_id.loc[triplet.positive_item_id]),
                doc_text(by_id.loc[triplet.negative_item_id]),
            ]
        )
        for triplet in triplets
        if triplet.positive_item_id in by_id.index and triplet.negative_item_id in by_id.index
    ]


def train(
    model: SentenceTransformer,
    triplets: list[HardNegativeTriplet],
    df: pd.DataFrame,
    *,
    output_dir: Path | str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> SentenceTransformer:
    """Fine-tune `model` (already LoRA-wrapped via `apply_lora`) on `triplets` with
    `MultipleNegativesRankingLoss`, and save it to `output_dir`."""
    examples = triplets_to_examples(triplets, df)
    if not examples:
        raise ValueError("No training examples produced — check that triplets reference ids present in df")

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    logger.info(
        "Training on %d triplets for %d epochs (batch_size=%d, lr=%s)",
        len(examples),
        epochs,
        batch_size,
        learning_rate,
    )
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=False,
    )
    return model


def run(
    *,
    products_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """End-to-end: load config + products, build training triplets, LoRA-wrap the base
    model, train, save to a versioned path, and return that path."""
    cfg = load_config()
    finetune_cfg = cfg["finetune"]

    products_path = (
        Path(products_path) if products_path else resolve_path(cfg["paths"]["products_enriched_parquet"])
    )
    df = pd.read_parquet(products_path)
    golden_cases = load_golden_set()
    triplets = build_training_triplets(df, golden_cases)

    base_model = SentenceTransformer(finetune_cfg["base_model"])
    model = apply_lora(base_model, finetune_cfg["lora"])

    output_dir = (
        Path(output_dir)
        if output_dir
        else resolve_path(cfg["paths"]["weights_dir"])
        / "finetuned"
        / f"{finetune_cfg['base_model'].split('/')[-1]}-lora-v1"
    )

    train(
        model,
        triplets,
        df,
        output_dir=output_dir,
        epochs=finetune_cfg["epochs"],
        batch_size=finetune_cfg["batch_size"],
        learning_rate=finetune_cfg["learning_rate"],
    )
    logger.info("Saved fine-tuned encoder to %s", output_dir)
    return output_dir


if __name__ == "__main__":
    run()
