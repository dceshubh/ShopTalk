"""Speech-to-text — `faster-whisper` wrapper (docs/ShopTalk_Plan.md Phase 8).

`faster-whisper` runs on CTranslate2, which supports CPU and CUDA but not Apple's MPS — so,
unlike the captioning/embedding modules, this one does NOT call `resolve_device()`. CPU with
`int8` quantization is the right call here anyway: Whisper-small is small enough that CPU
inference is already sub-second per utterance on an M3, and `int8` halves memory traffic
versus `float32` with no meaningful accuracy loss for short shopping queries.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Transcriber:
    """A loaded Whisper model, pinned to one device and compute type."""

    model: WhisperModel

    def transcribe(self, audio: str | Path | bytes) -> str:
        """Transcribe an audio file (path or raw bytes) to text.

        `faster-whisper` accepts a path, a file-like object, or a numpy array — bytes are
        wrapped in a `BytesIO` so callers can pass `st.audio_input` / file-upload payloads
        directly without writing a temp file.
        """
        source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
        segments, _info = self.model.transcribe(source, beam_size=1, language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()


def load_transcriber(model_name: str = "faster-whisper-small", *, device: str = "cpu") -> Transcriber:
    """Load the Whisper model named in `models.stt` (e.g. "faster-whisper-small" -> "small").

    Downloads and caches the CTranslate2-converted weights under the standard HF cache on
    first use (~500 MB for "small") — no manual model management, unlike Piper's voices.
    """
    size = model_name.removeprefix("faster-whisper-")
    logger.info("Loading faster-whisper model size=%s device=%s compute_type=int8", size, device)
    model = WhisperModel(size, device=device, compute_type="int8")
    return Transcriber(model=model)
