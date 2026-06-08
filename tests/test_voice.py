"""Unit tests for src.voice — the faster-whisper STT and Piper TTS wrappers
(docs/ShopTalk_Plan.md Phase 8). Model *loading* is faked throughout (downloading/loading
multi-hundred-MB weights has no place in a fast unit suite, same convention as
test_captioning.py); the wrapper logic around it — segment joining, bytes-vs-path handling,
WAV assembly — runs for real against fake-but-faithful stand-ins.
"""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import pytest

from src.voice.stt import Transcriber, load_transcriber
from src.voice.tts import Speaker, load_speaker

# ---------------------------------------------------------------------------
# stt.py — Transcriber / load_transcriber
# ---------------------------------------------------------------------------


def _fake_segment(text: str) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    return seg


def test_load_transcriber_strips_the_faster_whisper_prefix_and_uses_cpu_int8():
    with patch("src.voice.stt.WhisperModel") as mock_whisper_model:
        load_transcriber("faster-whisper-small")

    mock_whisper_model.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_transcribe_joins_segment_texts_into_one_trimmed_string():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (
        [_fake_segment(" show me "), _fake_segment("red dresses ")],
        MagicMock(),
    )

    transcriber = Transcriber(model=fake_model)

    assert transcriber.transcribe("some/path.wav") == "show me red dresses"
    fake_model.transcribe.assert_called_once_with("some/path.wav", beam_size=1, language="en")


def test_transcribe_wraps_raw_bytes_in_a_bytesio_so_uploads_need_no_temp_file():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([_fake_segment("hello")], MagicMock())

    transcriber = Transcriber(model=fake_model)
    transcriber.transcribe(b"raw-wav-bytes")

    (source,), kwargs = fake_model.transcribe.call_args
    assert isinstance(source, io.BytesIO)
    assert source.getvalue() == b"raw-wav-bytes"
    assert kwargs == {"beam_size": 1, "language": "en"}


# ---------------------------------------------------------------------------
# tts.py — Speaker / load_speaker
# ---------------------------------------------------------------------------


def test_load_speaker_raises_a_helpful_error_naming_the_download_command_when_voice_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m piper.download_voices"):
        load_speaker("en_US-lessac-low", voice_dir=tmp_path)


def test_load_speaker_loads_the_named_onnx_model_from_voice_dir(tmp_path):
    model_path = tmp_path / "en_US-lessac-low.onnx"
    model_path.write_bytes(b"fake-onnx-weights")

    with patch("src.voice.tts.PiperVoice.load") as mock_load:
        load_speaker("en_US-lessac-low", voice_dir=tmp_path)

    mock_load.assert_called_once_with(model_path)


def _fake_piper_voice(*, sample_rate: int, frames: bytes) -> MagicMock:
    """A minimal stand-in for `PiperVoice` whose `synthesize_wav` writes through the
    `wave.Wave_write` API exactly as the real Piper does — so `Speaker.synthesize`'s WAV
    assembly runs against real `wave` machinery, not a mock of it."""

    def _synthesize_wav(_text, wav_file, **_kwargs):
        wav_file.setframerate(sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(frames)
        return None

    voice = MagicMock()
    voice.synthesize_wav.side_effect = _synthesize_wav
    return voice


def test_synthesize_returns_playable_wav_bytes_with_the_voices_sample_rate():
    speaker = Speaker(voice=_fake_piper_voice(sample_rate=16000, frames=b"\x00\x01" * 100))

    wav_bytes = speaker.synthesize("Here are a few brown chairs you might like.")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1
        assert wav_file.readframes(wav_file.getnframes()) == b"\x00\x01" * 100
