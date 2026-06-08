"""Shared device-resolution — used by every module that loads a torch model (captioning,
embeddings, fine-tuning) so "dev on M3, run the heavy batch on a free cloud GPU" is a property
of one function, not re-implemented per module.
"""

from __future__ import annotations

import torch


def resolve_device() -> str:
    """Pick the best available device: CUDA (Kaggle/Colab T4) > MPS (Apple Silicon) > CPU.

    Keeping this device-agnostic is what lets the *same* module run a local dev sample on an
    M3 and the full-scale batch on a free cloud GPU — no fork in the code between "dev" and
    "real" runs, only a different `sample_size`.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
