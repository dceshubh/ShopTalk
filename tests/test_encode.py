"""Unit tests for src.embeddings.encode — the pluggable text-encoder wrapper behind the
"compare >= 3 pretrained encoders" requirement (`models.text_encoder` in config.yaml).

Model loading is faked throughout — pulling real SentenceTransformer weights has no place
in a fast unit suite. What we DO assert on is the part that's easy to get subtly wrong and
would quietly bias the comparison: that each registered model applies its OWN query/passage
instruction prefix, and only that one.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.embeddings.encode import Encoder, load_encoder


def _fake_st(*, encode_return=None):
    model = MagicMock()
    model.encode.return_value = encode_return if encode_return is not None else "fake-embeddings"
    return model


def test_load_encoder_rejects_unregistered_model_names():
    with pytest.raises(ValueError, match="Unknown text encoder"):
        load_encoder("not-a-real/model")


def test_load_encoder_wires_up_the_registered_prefix_convention():
    with patch("src.embeddings.encode.SentenceTransformer", return_value=_fake_st()):
        bge = load_encoder("BAAI/bge-base-en-v1.5", device="cpu")
        e5 = load_encoder("intfloat/e5-base-v2", device="cpu")
        minilm = load_encoder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    assert bge.query_prefix == "Represent this sentence for searching relevant passages: "
    assert bge.passage_prefix == ""

    assert e5.query_prefix == "query: "
    assert e5.passage_prefix == "passage: "

    assert minilm.query_prefix == ""
    assert minilm.passage_prefix == ""


def test_encode_queries_and_passages_apply_the_correct_side_specific_prefix():
    """The crux of a *fair* comparison: e5's query gets `query: `, its passage gets
    `passage: ` — mixing these up (or dropping them) silently handicaps the model relative
    to how it was contrastively pretrained.
    """
    model = _fake_st()
    encoder = Encoder(
        model_name="intfloat/e5-base-v2",
        model=model,
        device="cpu",
        query_prefix="query: ",
        passage_prefix="passage: ",
    )

    encoder.encode_queries(["a blue chair"])
    prefixed_queries = model.encode.call_args.args[0]
    assert prefixed_queries == ["query: a blue chair"]

    encoder.encode_passages(["Mid-Century Walnut Chair · color: Walnut"])
    prefixed_passages = model.encode.call_args.args[0]
    assert prefixed_passages == ["passage: Mid-Century Walnut Chair · color: Walnut"]


def test_encode_leaves_text_untouched_when_prefix_is_empty():
    """MiniLM (and bge's passage side) get no prefix — confirm we don't prepend an empty
    string and silently change every embedded string's leading whitespace/identity.
    """
    model = _fake_st()
    encoder = Encoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model=model,
        device="cpu",
        query_prefix="",
        passage_prefix="",
    )

    encoder.encode_queries(["a blue chair"])
    assert model.encode.call_args.args[0] == ["a blue chair"]


def test_encode_normalizes_embeddings_for_cosine_as_dot_product():
    model = _fake_st()
    encoder = Encoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model=model,
        device="cpu",
        query_prefix="",
        passage_prefix="",
    )

    encoder.encode_passages(["a wooden chair"])

    _, kwargs = model.encode.call_args
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True


def test_load_encoder_auto_resolves_device_when_not_given():
    with (
        patch("src.embeddings.encode.SentenceTransformer", return_value=_fake_st()) as mock_st,
        patch("src.embeddings.encode.resolve_device", return_value="mps"),
    ):
        encoder = load_encoder("sentence-transformers/all-MiniLM-L6-v2")

    assert encoder.device == "mps"
    mock_st.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="mps")
