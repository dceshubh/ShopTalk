"""Tests for src.ui.app — the Streamlit chat UI (docs/ShopTalk_Plan.md Phase 7).

Driven via `streamlit.testing.v1.AppTest`, which re-executes `app.py` as a fresh script on
every `.run()` — so module-level monkeypatches on an imported `src.ui.app` do NOT reach the
running script. The one expensive edge (the network call to the FastAPI backend) is faked at
the `httpx.post` boundary instead, which IS shared across the re-exec; everything else
(sidebar wiring, session state, message rendering, feedback buttons, real SQLite store) runs
for real, in keeping with this project's "fake the expensive edge, test the wiring"
convention (see tests/test_api.py, tests/test_graph.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from src.ui.app import _apply_sidebar_filters
from src.ui.feedback import load_feedback_store

_APP_PATH = "src/ui/app.py"


def _fake_chat_response(response_text: str, products: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = lambda: None
    response.json = lambda: {"response_text": response_text, "products": products}
    return response


# ---------------------------------------------------------------------------
# _apply_sidebar_filters — pure function, no Streamlit runtime needed
# ---------------------------------------------------------------------------


def test_apply_sidebar_filters_passes_message_through_when_everything_is_any():
    assert (
        _apply_sidebar_filters("show me chairs", product_type="Any", color="Any", material="Any")
        == "show me chairs"
    )


def test_apply_sidebar_filters_folds_non_any_picks_into_the_message_text():
    out = _apply_sidebar_filters("show me chairs", product_type="CHAIR", color="Brown", material="Any")

    assert out.startswith("show me chairs")
    assert "CHAIR" in out
    assert "Brown" in out
    assert "Any" not in out


# ---------------------------------------------------------------------------
# Initial render — sidebar, identity, layout
# ---------------------------------------------------------------------------


def test_app_renders_title_and_sidebar_controls_on_first_load():
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "🛍️ ShopTalk"
    assert len(at.text_input) == 1  # the user_id field
    assert at.text_input[0].value.startswith("user-")  # stable random identity, no login
    assert [box.label for box in at.selectbox] == ["Product type", "Color", "Material"]
    assert all(box.value == "Any" for box in at.selectbox)
    assert len(at.chat_message) == 0  # no history yet


def test_user_id_is_a_stable_editable_text_field_not_a_login_flow():
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    at.text_input[0].set_value("shubham").run(timeout=30)

    assert not at.exception
    assert at.session_state["user_id"] == "shubham"
    assert at.text_input[0].value == "shubham"


def test_new_conversation_button_resets_session_and_history():
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    original_session_id = at.session_state["session_id"]

    new_conv_button = next(b for b in at.button if b.label == "New conversation")
    new_conv_button.click().run(timeout=30)

    assert not at.exception
    assert at.session_state["session_id"] != original_session_id
    assert at.session_state["messages"] == []


# ---------------------------------------------------------------------------
# Chat round trip — backend faked at httpx.post, everything else real
# ---------------------------------------------------------------------------


def test_sending_a_message_renders_response_text_and_product_cards_with_feedback_buttons():
    fake = _fake_chat_response(
        "Here are a few brown chairs you might like!",
        [
            {
                "item_id": "B0TEST0001",
                "name": "Cozy Reading Chair",
                "image_path": "c0/c096fa8d.jpg",
                "product_type": "CHAIR",
                "color": "Brown",
                "material": "Wood",
                "brand": "Acme",
            }
        ],
    )

    with patch("httpx.post", return_value=fake) as mock_post:
        at = AppTest.from_file(_APP_PATH)
        at.run(timeout=30)
        at.chat_input[0].set_value("show me brown chairs").run(timeout=30)

    assert not at.exception
    assert mock_post.call_args.kwargs["json"]["message"] == "show me brown chairs"

    roles = [m.name for m in at.chat_message]
    assert roles == ["user", "assistant"]
    assert at.chat_message[0].markdown[0].value == "show me brown chairs"

    assistant = at.chat_message[1]
    assert any("Here are a few brown chairs" in m.value for m in assistant.markdown)
    assert any("Cozy Reading Chair" in m.value for m in assistant.markdown)

    button_labels = sorted(b.label for b in assistant.button)
    assert button_labels == ["👍", "👎"]


def test_thumbs_up_persists_a_verdict_to_the_real_feedback_store(tmp_path):
    """`AppTest` re-execs `app.py` as a fresh script per `.run()`, so a patch on the
    already-imported `src.ui.app` module wouldn't reach it. `src.ui.feedback` IS shared
    (it stays in `sys.modules`), and `app.py`'s `from ... import load_feedback_store`
    re-reads that module attribute on every fresh exec — so patching it there propagates.
    `st.cache_resource` is ALSO a process-global singleton cache keyed by function
    identity — clear it first, or an earlier test's cached store (pointed at a different
    tmp_path) wins regardless of the patch below."""
    import streamlit as st

    st.cache_resource.clear()
    db_path = tmp_path / "feedback.sqlite"

    fake = _fake_chat_response(
        "Here's a sturdy boot.",
        [
            {
                "item_id": "B0TEST0002",
                "name": "Trail Boot",
                "image_path": None,
                "product_type": "BOOT",
                "color": "Black",
                "material": "Leather",
                "brand": "Trekker",
            }
        ],
    )

    with (
        patch("httpx.post", return_value=fake),
        patch("src.ui.feedback.load_feedback_store", lambda *a, **kw: load_feedback_store(db_path)),
    ):
        at = AppTest.from_file(_APP_PATH)
        at.run(timeout=30)
        at.chat_input[0].set_value("rugged boots please").run(timeout=30)

        assistant = at.chat_message[1]
        up_button = next(b for b in assistant.button if b.label == "👍")
        up_button.click().run(timeout=30)

    assert not at.exception
    store = load_feedback_store(db_path)
    assert (
        store.verdict_for(
            user_id=at.session_state["user_id"], query="rugged boots please", item_id="B0TEST0002"
        )
        == "up"
    )


# ---------------------------------------------------------------------------
# Voice mode (docs/ShopTalk_Plan.md Phase 8) — toggle + TTS wiring.
#
# `AppTest` (pinned streamlit==1.39.0) has no proxy for simulating a `file_uploader`
# upload, so the STT half (upload -> transcribe -> prompt) isn't exercisable from this
# suite — it's covered at the wrapper level by tests/test_voice.py and was live-tested
# end-to-end (see docs/ShopTalk_Plan.md Phase 8 exit gates). The TTS half needs no such
# simulation: it fires unconditionally once a chat turn completes with voice mode on, so
# the same `httpx.post` + chat_input drive used elsewhere reaches it directly.
# ---------------------------------------------------------------------------


def test_voice_mode_checkbox_reveals_an_audio_upload_control():
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    voice_checkbox = next(c for c in at.checkbox if c.label == "🎙️ Voice mode")
    assert voice_checkbox.value is False
    elements_before = len(at.main.children)

    voice_checkbox.set_value(True).run(timeout=30)

    assert not at.exception
    assert len(at.main.children) > elements_before  # the file_uploader rendered


def test_voice_mode_speaks_the_response_and_stores_the_audio_with_the_turn():
    """`AppTest` re-execs `app.py` per `.run()`, so — same gotcha as the feedback-store
    test — a patch on the already-imported `src.ui.app` module wouldn't reach the running
    script. `src.voice.tts.load_speaker` IS shared across the re-exec (the module stays in
    `sys.modules`), and `app.py`'s `_speaker()` re-imports it fresh on every exec via
    `from src.voice.tts import ... load_speaker`."""
    fake = _fake_chat_response(
        "Here's a sturdy boot.",
        [
            {
                "item_id": "B0TEST0003",
                "name": "Trail Boot",
                "image_path": None,
                "product_type": "BOOT",
                "color": "Black",
                "material": "Leather",
                "brand": "Trekker",
            }
        ],
    )
    fake_audio = b"RIFF....WAVEfmt fake-pcm-bytes"
    fake_speaker = MagicMock()
    fake_speaker.synthesize.return_value = fake_audio

    with (
        patch("httpx.post", return_value=fake),
        patch("src.voice.tts.load_speaker", lambda *a, **kw: fake_speaker),
    ):
        at = AppTest.from_file(_APP_PATH)
        at.run(timeout=30)

        voice_checkbox = next(c for c in at.checkbox if c.label == "🎙️ Voice mode")
        voice_checkbox.set_value(True).run(timeout=30)
        at.chat_input[0].set_value("rugged boots please").run(timeout=30)

    assert not at.exception
    fake_speaker.synthesize.assert_called_once_with("Here's a sturdy boot.")

    assistant_message = at.session_state["messages"][-1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["audio"] == fake_audio


def test_unreachable_backend_surfaces_a_friendly_error_instead_of_crashing():
    import httpx as httpx_module

    with patch("httpx.post", side_effect=httpx_module.ConnectError("connection refused")):
        at = AppTest.from_file(_APP_PATH)
        at.run(timeout=30)
        at.chat_input[0].set_value("anything").run(timeout=30)

    assert not at.exception
    assistant = at.chat_message[1]
    assert any("Couldn't reach ShopTalk's backend" in e.value for e in assistant.error)
