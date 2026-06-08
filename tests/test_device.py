"""Tests for src.common.device — the shared device-resolution every torch-loading module
(captioning, embeddings, fine-tuning) relies on for "dev on M3, run the heavy batch on a free
cloud GPU" without forking code paths.
"""

from __future__ import annotations

from unittest.mock import patch

from src.common.device import resolve_device


def test_resolve_device_prefers_cuda_then_mps_then_cpu():
    with patch("src.common.device.torch.cuda.is_available", return_value=True):
        assert resolve_device() == "cuda"

    with (
        patch("src.common.device.torch.cuda.is_available", return_value=False),
        patch("src.common.device.torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device() == "mps"

    with (
        patch("src.common.device.torch.cuda.is_available", return_value=False),
        patch("src.common.device.torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device() == "cpu"
