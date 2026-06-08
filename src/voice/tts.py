"""Text-to-speech — Piper wrapper (docs/ShopTalk_Plan.md Phase 8).

Piper is a fully local, CPU-friendly neural TTS engine — `piper-tts==1.4.2` ships proper
`cp39-abi3` arm64 wheels (verified on this machine; the project's original
`requirements.txt` note claiming it was uninstallable on Python 3.12/arm64 referred to an
older release whose `piper-phonemize` dependency had no such wheel). A voice is two files —
an ONNX model (`<voice>.onnx`) and its JSON config (`<voice>.onnx.json`) — downloaded once via
`python -m piper.download_voices` into `paths.piper_voice_dir` and reused across runs (not
committed to git: see `weights/` in `.gitignore`).
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

from piper import PiperVoice

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Speaker:
    """A loaded Piper voice that renders text to in-memory WAV bytes."""

    voice: PiperVoice

    def synthesize(self, text: str) -> bytes:
        """Render `text` to a mono 16-bit PCM WAV byte string, ready for `st.audio`.

        Piper streams audio one sentence-chunk at a time via `synthesize_wav` — buffering
        into a `BytesIO`-backed `wave.Wave_write` keeps everything in memory (no temp files,
        the same "no incidental disk state" property the rest of the pipeline holds to).
        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()


def load_speaker(voice_name: str, *, voice_dir: str | Path) -> Speaker:
    """Load the Piper voice named `voice_name` (e.g. "en_US-lessac-low") from `voice_dir`.

    Expects `<voice_dir>/<voice_name>.onnx` (+ `.onnx.json`) to already exist — fetch them
    once via `python -m piper.download_voices --download-dir <voice_dir> <voice_name>`
    (see README "Voice mode" setup). Raising early with that exact command beats a confusing
    `FileNotFoundError` deep inside `PiperVoice.load`.
    """
    model_path = Path(voice_dir) / f"{voice_name}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper voice {voice_name!r} not found at {model_path}. Download it once with:\n"
            f"  python -m piper.download_voices --download-dir {voice_dir} {voice_name}"
        )
    logger.info("Loading Piper voice %s from %s", voice_name, model_path)
    voice = PiperVoice.load(model_path)
    return Speaker(voice=voice)
