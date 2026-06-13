"""Unit tests for src.embeddings.finetune.trainer — LoRA wiring and the training-loop
plumbing for Phase 4 (docs/ShopTalk_Plan.md §2.3).

Loading real model weights and actually training has no place in a fast unit suite (same
convention as tests/test_encode.py). What's under test: `apply_lora` builds a `LoraConfig`
from `configs/config.yaml`'s `finetune.lora` values and wraps the right attribute;
`triplets_to_examples` resolves triplets to `InputExample`s via the shared `doc_text`;
`train` calls `model.fit` with the configured hyperparameters; `run` wires config -> data ->
model -> train end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.embeddings.finetune.trainer import (
    LORA_TARGET_MODULES,
    apply_lora,
    run,
    train,
    triplets_to_examples,
)
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
        ]
    )


# ---------------------------------------------------------------------------
# apply_lora
# ---------------------------------------------------------------------------


def test_apply_lora_builds_lora_config_from_finetune_config_and_wraps_auto_model():
    fake_model = MagicMock()
    fake_backbone = fake_model.__getitem__.return_value
    fake_backbone.auto_model = "original-auto-model"
    lora_cfg = {"r": 16, "alpha": 32, "dropout": 0.1}

    with (
        patch("src.embeddings.finetune.trainer.LoraConfig") as mock_lora_config,
        patch(
            "src.embeddings.finetune.trainer.get_peft_model", return_value="peft-wrapped-model"
        ) as mock_get_peft_model,
    ):
        result = apply_lora(fake_model, lora_cfg)

    mock_lora_config.assert_called_once_with(
        r=16, lora_alpha=32, lora_dropout=0.1, target_modules=LORA_TARGET_MODULES, bias="none"
    )
    mock_get_peft_model.assert_called_once_with("original-auto-model", mock_lora_config.return_value)
    assert fake_backbone.auto_model == "peft-wrapped-model"
    assert result is fake_model


# ---------------------------------------------------------------------------
# triplets_to_examples
# ---------------------------------------------------------------------------


def test_triplets_to_examples_resolves_ids_to_doc_text():
    df = _products_df()
    triplets = [HardNegativeTriplet(query="walnut table", positive_item_id="POS1", negative_item_id="NEG1")]

    [example] = triplets_to_examples(triplets, df)

    assert example.texts[0] == "walnut table"
    assert "Walnut Coffee Table" in example.texts[1]
    assert "Teal Side Table" in example.texts[2]


def test_triplets_to_examples_drops_triplets_with_unknown_ids():
    df = _products_df()
    triplets = [HardNegativeTriplet(query="mystery", positive_item_id="NOT_IN_DF", negative_item_id="NEG1")]

    assert triplets_to_examples(triplets, df) == []


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_raises_when_there_are_no_usable_examples():
    df = _products_df()
    model = MagicMock()

    with pytest.raises(ValueError, match="No training examples"):
        train(model, [], df, output_dir="/tmp/out", epochs=1, batch_size=8, learning_rate=1e-5)


def test_train_calls_model_fit_with_configured_hyperparameters():
    df = _products_df()
    model = MagicMock()
    triplets = [HardNegativeTriplet(query="walnut table", positive_item_id="POS1", negative_item_id="NEG1")]

    with patch(
        "src.embeddings.finetune.trainer.losses.MultipleNegativesRankingLoss", return_value="loss-fn"
    ) as mock_loss:
        train(model, triplets, df, output_dir="/tmp/out", epochs=3, batch_size=8, learning_rate=2e-5)

    mock_loss.assert_called_once_with(model)
    model.fit.assert_called_once()
    _, kwargs = model.fit.call_args
    assert kwargs["epochs"] == 3
    assert kwargs["optimizer_params"] == {"lr": 2e-5}
    assert kwargs["output_path"] == "/tmp/out"
    [(loader, loss_fn)] = kwargs["train_objectives"]
    assert loss_fn == "loss-fn"
    assert loader.batch_size == 8


# ---------------------------------------------------------------------------
# run (end-to-end wiring)
# ---------------------------------------------------------------------------


def test_run_wires_config_data_model_and_train_together(tmp_path):
    df = _products_df()
    fake_config = {
        "paths": {"products_enriched_parquet": "ignored", "weights_dir": str(tmp_path)},
        "finetune": {
            "base_model": "BAAI/bge-base-en-v1.5",
            "lora": {"r": 16, "alpha": 32, "dropout": 0.1},
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 2e-5,
        },
    }
    triplets = [HardNegativeTriplet(query="walnut table", positive_item_id="POS1", negative_item_id="NEG1")]

    with (
        patch("src.embeddings.finetune.trainer.load_config", return_value=fake_config),
        patch("src.embeddings.finetune.trainer.pd.read_parquet", return_value=df),
        patch("src.embeddings.finetune.trainer.load_golden_set", return_value=[]),
        patch("src.embeddings.finetune.trainer.build_training_triplets", return_value=triplets),
        patch("src.embeddings.finetune.trainer.SentenceTransformer", return_value="base-model"),
        patch("src.embeddings.finetune.trainer.apply_lora", return_value=MagicMock()) as mock_apply_lora,
        patch("src.embeddings.finetune.trainer.train") as mock_train,
    ):
        output_dir = run(products_path="ignored")

    mock_apply_lora.assert_called_once_with("base-model", fake_config["finetune"]["lora"])
    mock_train.assert_called_once()
    _, kwargs = mock_train.call_args
    assert kwargs["epochs"] == 1
    assert kwargs["batch_size"] == 8
    assert kwargs["learning_rate"] == 2e-5
    assert output_dir == tmp_path / "finetuned" / "bge-base-en-v1.5-lora-v1"
