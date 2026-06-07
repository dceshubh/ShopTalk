"""Unit tests for the captioning pipeline — image fetching/caching, model-loading
contracts, and the caption -> doc-rebuild handoff. Network calls and model loading are
faked throughout: hitting real S3 or loading multi-GB weights has no place in a fast unit
suite. The actual BLIP-2 vs BLIP-base qualitative comparison is a manual-review exit-gate
step (see docs/ShopTalk_Plan.md Phase 2), not something a unit test can assert on.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.captioning.caption import Captioner, load_captioner, resolve_device
from src.captioning.enrich import caption_products
from src.captioning.images import ensure_images_cached, fetch_image

# ---------------------------------------------------------------------------
# images.py — fetch / cache
# ---------------------------------------------------------------------------


def test_fetch_image_skips_download_when_already_cached(tmp_path):
    cached = tmp_path / "ab" / "abc123.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"fake-image-bytes")

    with patch("src.captioning.images.httpx.get") as mock_get:
        result = fetch_image("ab/abc123.jpg", tmp_path)

    mock_get.assert_not_called()
    assert result == cached


def test_fetch_image_downloads_to_the_documented_s3_path_and_caches_on_miss(tmp_path):
    mock_response = MagicMock(content=b"downloaded-bytes")
    mock_response.raise_for_status.return_value = None

    with patch("src.captioning.images.httpx.get", return_value=mock_response) as mock_get:
        result = fetch_image("cd/def456.jpg", tmp_path)

    mock_get.assert_called_once()
    requested_url = mock_get.call_args.args[0]
    assert requested_url == "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/cd/def456.jpg"
    assert result == tmp_path / "cd" / "def456.jpg"
    assert result.read_bytes() == b"downloaded-bytes"


def test_ensure_images_cached_resolves_every_path_and_reuses_the_cache(tmp_path):
    mock_response = MagicMock(content=b"x")
    mock_response.raise_for_status.return_value = None

    with patch("src.captioning.images.httpx.get", return_value=mock_response) as mock_get:
        resolved = ensure_images_cached(["a/1.jpg", "a/1.jpg", "b/2.jpg"], tmp_path)

    assert set(resolved) == {"a/1.jpg", "b/2.jpg"}
    # The repeated "a/1.jpg" hits the on-disk cache fetch_image just populated — one GET
    # per *unique* path, even though the helper doesn't dedupe its input list itself.
    assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# caption.py — device resolution, registry, generation contract
# ---------------------------------------------------------------------------


def test_resolve_device_prefers_cuda_then_mps_then_cpu():
    with patch("src.captioning.caption.torch.cuda.is_available", return_value=True):
        assert resolve_device() == "cuda"

    with (
        patch("src.captioning.caption.torch.cuda.is_available", return_value=False),
        patch("src.captioning.caption.torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device() == "mps"

    with (
        patch("src.captioning.caption.torch.cuda.is_available", return_value=False),
        patch("src.captioning.caption.torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device() == "cpu"


def test_load_captioner_rejects_unregistered_model_names():
    with pytest.raises(ValueError, match="Unknown caption model"):
        load_captioner("not-a-real/model")


def test_captioner_caption_strips_and_decodes_generated_ids():
    fake_tensor = MagicMock()
    fake_tensor.to.return_value = fake_tensor
    processor = MagicMock(return_value={"pixel_values": fake_tensor})
    processor.batch_decode.return_value = ["  a red sneaker on a white background  "]

    model = MagicMock(dtype="float32")
    model.generate.return_value = "generated-ids"

    captioner = Captioner(model_name="fake/model", processor=processor, model=model, device="cpu")
    caption = captioner.caption(image=MagicMock(), max_new_tokens=30)

    assert caption == "a red sneaker on a white background"
    model.generate.assert_called_once_with(pixel_values=fake_tensor, max_new_tokens=30)
    processor.batch_decode.assert_called_once_with("generated-ids", skip_special_tokens=True)


# ---------------------------------------------------------------------------
# enrich.py — caption_products: success/failure handling, shared-transformer doc rebuild
# ---------------------------------------------------------------------------


def _products_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "item_id": "X",
        "domain_name": "amazon.com",
        "product_type": "CHAIR",
        "name": "Mid-Century Walnut Chair",
        "brand": "Rivet",
        "color": "Walnut",
        "material": "Wood",
        "bullet_points": ["Solid wood frame"],
        "keywords": ["chair"],
        "main_image_id": "IMG1",
        "image_path": "ab/img1.jpg",
        "doc_text": "placeholder — caption_products always rebuilds this",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _open_unless_path_contains(marker: str):
    """Build a fake `Image.open` that raises for paths containing `marker`, succeeds otherwise."""

    def _fake_open(path):
        if marker in str(path):
            raise OSError("simulated corrupt image")
        return MagicMock(convert=MagicMock(return_value="rgb-image"))

    return _fake_open


def test_caption_products_appends_visual_segment_via_shared_build_doc_text(tmp_path):
    df = _products_df([{"item_id": "P1", "image_path": "aa/p1.jpg"}])
    local_paths = {"aa/p1.jpg": tmp_path / "aa" / "p1.jpg"}

    captioner = MagicMock(model_name="fake/model")
    captioner.caption.return_value = "a marble-pattern phone case"

    with (
        patch("src.captioning.enrich.ensure_images_cached", return_value=local_paths),
        patch("src.captioning.enrich.Image.open", side_effect=_open_unless_path_contains("__never__")),
    ):
        enriched = caption_products(df, captioner, tmp_path, max_new_tokens=30)

    row = enriched.iloc[0]
    assert row["visual_caption"] == "a marble-pattern phone case"
    # `build_doc_text` (the same function the offline indexer / online API call) appends
    # the visual segment last — confirmed in tests/test_preprocess.py.
    assert row["doc_text"].endswith("visual: a marble-pattern phone case")
    assert "Mid-Century Walnut Chair" in row["doc_text"]


def test_caption_products_keeps_failed_items_with_no_visual_segment(tmp_path):
    df = _products_df(
        [
            {"item_id": "OK1", "image_path": "aa/ok1.jpg"},
            {"item_id": "BAD", "image_path": "bb/bad.jpg"},
            {"item_id": "OK2", "image_path": "cc/ok2.jpg"},
        ]
    )
    local_paths = {p: tmp_path / p for p in df["image_path"]}

    captioner = MagicMock(model_name="fake/model")
    captioner.caption.return_value = "a wooden chair with armrests"

    with (
        patch("src.captioning.enrich.ensure_images_cached", return_value=local_paths),
        patch("src.captioning.enrich.Image.open", side_effect=_open_unless_path_contains("bad")),
    ):
        enriched = caption_products(df, captioner, tmp_path, max_new_tokens=30)

    by_id = enriched.set_index("item_id")

    assert by_id.loc["BAD", "visual_caption"] is None
    assert "visual:" not in by_id.loc["BAD", "doc_text"]

    for ok_id in ("OK1", "OK2"):
        assert by_id.loc[ok_id, "visual_caption"] == "a wooden chair with armrests"
        assert by_id.loc[ok_id, "doc_text"].endswith("visual: a wooden chair with armrests")

    # Every row keeps a non-empty doc_text — captioning failures degrade to text-only,
    # they never blank out the document (that would silently break retrieval for the item).
    assert enriched["doc_text"].str.strip().str.len().gt(0).all()


def test_caption_products_dedupes_image_fetch_requests(tmp_path):
    df = _products_df(
        [
            {"item_id": "A", "image_path": "shared.jpg"},
            {"item_id": "B", "image_path": "shared.jpg"},
        ]
    )
    captioner = MagicMock(model_name="fake/model")
    captioner.caption.return_value = "x"

    with (
        patch(
            "src.captioning.enrich.ensure_images_cached",
            return_value={"shared.jpg": tmp_path / "shared.jpg"},
        ) as mock_ensure,
        patch("src.captioning.enrich.Image.open", side_effect=_open_unless_path_contains("__never__")),
    ):
        caption_products(df, captioner, tmp_path)

    # Order-preserving de-dupe: the two rows share one image, so exactly one path is fetched.
    mock_ensure.assert_called_once_with(["shared.jpg"], tmp_path)
