"""ShopTalk — Streamlit chat UI (docs/ShopTalk_Plan.md Phase 7).

Talks to the FastAPI backend over HTTP (`ui.api_base_url`) — never in-process — so the UI
and the deployment surface stay cleanly separated and the API's "models loaded once, same
transformers" properties aren't accidentally bypassed by a shortcut import.

Run with: `streamlit run src/ui/app.py` (the FastAPI service must be running separately).
"""

from __future__ import annotations

import uuid

import httpx
import streamlit as st

from src.common.config import load_config, resolve_path
from src.ui.feedback import FeedbackStore, load_feedback_store
from src.voice.stt import Transcriber, load_transcriber
from src.voice.tts import Speaker, load_speaker

# A curated subset of common attribute values for the sidebar filter UI — NOT the full
# catalog vocabulary (that lives only in the API/index). "Any" means "let the agent's own
# conversational filter-extraction decide" — these are optional nudges, not a parallel
# filter pipeline.
_PRODUCT_TYPES = ["Any", "CHAIR", "SOFA", "BOOT", "SHOES", "RUG", "LAMP", "BED", "TABLE"]
_COLORS = ["Any", "Black", "Brown", "Blue", "White", "Grey", "Beige", "Red", "Green", "Pink"]
_MATERIALS = ["Any", "Leather", "Wood", "Metal", "Plastic", "Fabric", "Cotton", "Glass"]


def _config():
    cfg = load_config()
    return cfg["ui"]["api_base_url"], cfg["ui"]["title"]


@st.cache_resource
def _feedback_store() -> FeedbackStore:
    return load_feedback_store()


@st.cache_resource
def _transcriber() -> Transcriber:
    """Loaded once per process — Whisper-small (~500 MB, CPU/int8) is too heavy to reload
    on every rerun (`st.cache_resource` is the same singleton-cache pattern as `_feedback_store`)."""
    return load_transcriber(load_config()["models"]["stt"])


@st.cache_resource
def _speaker() -> Speaker:
    """Loaded once per process from the local Piper voice files — see `models.tts_voice`
    and `paths.piper_voice_dir` in config.yaml, and the README "Voice mode" setup for how
    those files get there (`python -m piper.download_voices`, a one-time ~60 MB download)."""
    cfg = load_config()
    return load_speaker(cfg["models"]["tts_voice"], voice_dir=resolve_path(cfg["paths"]["piper_voice_dir"]))


def _init_session_state() -> None:
    st.session_state.setdefault("user_id", f"user-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("session_id", f"session-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("messages", [])  # [{role, content, products?, query?}]


def _apply_sidebar_filters(message: str, *, product_type: str, color: str, material: str) -> str:
    """Fold any non-"Any" sidebar selections into the message as a natural-language
    qualifier — the SAME `extract_filters` LLM call that powers conversational filtering
    picks these up, so there is exactly one filter-extraction code path, not two."""
    picks = [v for v in (product_type, color, material) if v != "Any"]
    if not picks:
        return message
    return f"{message} (looking specifically for: {', '.join(picks)})"


def _call_chat_api(api_base_url: str, *, user_id: str, session_id: str, message: str) -> dict:
    response = httpx.post(
        f"{api_base_url}/chat",
        json={"user_id": user_id, "session_id": session_id, "message": message},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def _render_product_cards(products: list[dict], *, api_base_url: str, query: str, msg_key: str) -> None:
    if not products:
        st.caption("No matching products were retrieved for this turn.")
        return

    store = _feedback_store()
    user_id = st.session_state.user_id
    columns = st.columns(min(len(products), 3))
    for i, product in enumerate(products):
        item_id = product["item_id"]
        with columns[i % len(columns)]:
            if product.get("image_path"):
                st.image(f"{api_base_url}/images/{product['image_path']}", use_column_width=True)
            st.markdown(f"**{product.get('name') or item_id}**")
            attrs = ", ".join(
                str(v)
                for v in (
                    product.get("product_type"),
                    product.get("color"),
                    product.get("material"),
                    product.get("brand"),
                )
                if v
            )
            if attrs:
                st.caption(attrs)
            st.link_button("View product ↗", f"{api_base_url}/products/{item_id}", help=item_id)
            st.caption(f"id: `{item_id}`")

            current_verdict = store.verdict_for(user_id=user_id, query=query, item_id=item_id)
            up_col, down_col = st.columns(2)
            with up_col:
                if st.button("👍" + (" ✓" if current_verdict == "up" else ""), key=f"{msg_key}-up-{item_id}"):
                    store.record(
                        user_id=user_id,
                        session_id=st.session_state.session_id,
                        query=query,
                        item_id=item_id,
                        verdict="up",
                    )
                    st.rerun()
            with down_col:
                if st.button(
                    "👎" + (" ✓" if current_verdict == "down" else ""), key=f"{msg_key}-down-{item_id}"
                ):
                    store.record(
                        user_id=user_id,
                        session_id=st.session_state.session_id,
                        query=query,
                        item_id=item_id,
                        verdict="down",
                    )
                    st.rerun()


def main() -> None:
    api_base_url, title = _config()
    st.set_page_config(page_title=title, page_icon="🛍️", layout="wide")
    _init_session_state()

    with st.sidebar:
        st.header(title)
        st.text_input(
            "Your ID",
            key="user_id",
            help="A stable id — long-term preferences (e.g. favorite colors) persist across sessions under this id.",
        )
        st.caption(f"Session: `{st.session_state.session_id}`")
        st.divider()
        st.subheader("Filters (optional)")
        product_type = st.selectbox("Product type", _PRODUCT_TYPES)
        color = st.selectbox("Color", _COLORS)
        material = st.selectbox("Material", _MATERIALS)
        st.caption(
            "Selections are blended into your next message — ShopTalk's agent extracts them conversationally, same as if you'd typed them."
        )
        st.divider()
        voice_mode = st.checkbox(
            "🎙️ Voice mode",
            value=False,
            help="Upload a short voice clip instead of typing — ShopTalk transcribes it "
            "(faster-whisper, runs locally) and speaks its reply back (Piper TTS, also local).",
        )
        st.divider()
        if st.button("New conversation"):
            st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.rerun()

    st.title(f"🛍️ {title}")
    st.caption("Ask for anything — I'll search the catalog and remember what you like.")

    # Each assistant turn carries its own `turn_id`, generated once at append-time and
    # reused as the feedback-button key prefix on every subsequent rerun. Without this,
    # a card rendered "live" (during the turn that produced it) and the SAME card rendered
    # later from `st.session_state.messages` would get different positional keys — and a
    # 👍/👎 click would trigger a rerun that re-renders the card under a new key, silently
    # dropping the click before `FeedbackStore.record` ever runs.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/wav")
            if message.get("products") is not None:
                _render_product_cards(
                    message["products"],
                    api_base_url=api_base_url,
                    query=message.get("query", ""),
                    msg_key=f"turn-{message['turn_id']}",
                )

    prompt = st.chat_input("What are you shopping for?")

    # Voice input: an uploaded clip is transcribed once (guarded by `file_id` so a rerun
    # — e.g. clicking 👍 on a card — doesn't resubmit the same clip as a brand-new turn)
    # and treated exactly like a typed `chat_input` value from here on — one query path,
    # not two, same principle as `_apply_sidebar_filters` folding picks into the message text.
    if voice_mode:
        audio_upload = st.file_uploader(
            "Or upload a voice clip (wav / mp3 / m4a / ogg)",
            type=["wav", "mp3", "m4a", "ogg"],
            key="voice_upload",
        )
        if audio_upload is not None and audio_upload.file_id != st.session_state.get("_last_voice_upload_id"):
            st.session_state["_last_voice_upload_id"] = audio_upload.file_id
            with st.spinner("Transcribing..."):
                transcribed = _transcriber().transcribe(audio_upload.getvalue())
            st.caption(f"🎙️ Heard: “{transcribed}”")
            if transcribed:
                prompt = transcribed

    if prompt:
        turn_id = uuid.uuid4().hex[:8]
        st.session_state.messages.append({"role": "user", "content": prompt, "turn_id": turn_id})
        with st.chat_message("user"):
            st.write(prompt)

        outgoing = _apply_sidebar_filters(prompt, product_type=product_type, color=color, material=material)
        turn_id = uuid.uuid4().hex[:8]
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the catalog..."):
                    result = _call_chat_api(
                        api_base_url,
                        user_id=st.session_state.user_id,
                        session_id=st.session_state.session_id,
                        message=outgoing,
                    )
            except httpx.HTTPError as exc:
                st.error(f"Couldn't reach ShopTalk's backend at {api_base_url} — is the API running? ({exc})")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ Backend unreachable: {exc}",
                        "products": None,
                        "turn_id": turn_id,
                    }
                )
                return

            st.write(result["response_text"])
            audio_bytes = None
            if voice_mode:
                with st.spinner("Generating voice reply..."):
                    audio_bytes = _speaker().synthesize(result["response_text"])
                st.audio(audio_bytes, format="audio/wav")
            _render_product_cards(
                result["products"], api_base_url=api_base_url, query=prompt, msg_key=f"turn-{turn_id}"
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response_text"],
                "products": result["products"],
                "query": prompt,
                "turn_id": turn_id,
                "audio": audio_bytes,
            }
        )


if __name__ == "__main__":
    main()
