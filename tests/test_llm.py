"""Unit tests for src.agent.llm — the Groq-hosted generator-LLM wrapper. The Groq client
is faked throughout (a real call costs money and needs GROQ_API_KEY); what we assert on is
the part most likely to be silently wrong: that `complete_structured` actually constrains
the model to the target Pydantic schema (via JSON mode + an injected schema description)
and validates the response into a real Pydantic instance rather than handing back raw text.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.agent.llm import GeneratorLLM, load_generator


class _Filters(BaseModel):
    color: str | None = None
    max_price: float | None = None


def _fake_groq_client(content: str):
    client = MagicMock()
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ---------------------------------------------------------------------------
# load_generator
# ---------------------------------------------------------------------------


def test_load_generator_requires_api_key_and_defaults_to_configured_model():
    missing_key_error = RuntimeError(
        "Missing required environment variable 'GROQ_API_KEY' — copy .env.example to .env and fill it in."
    )
    with (
        patch("src.agent.llm.get_env", side_effect=missing_key_error),
        pytest.raises(RuntimeError, match="GROQ_API_KEY"),
    ):
        load_generator()


def test_load_generator_wires_the_api_key_into_the_groq_client():
    with (
        patch("src.agent.llm.get_env", return_value="fake-key"),
        patch("src.agent.llm.Groq") as mock_groq_cls,
    ):
        generator = load_generator("llama-3.1-8b-instant")

    mock_groq_cls.assert_called_once_with(api_key="fake-key")
    assert generator.model_name == "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# complete — free-text chat
# ---------------------------------------------------------------------------


def test_complete_returns_the_message_content_and_forwards_model_and_messages():
    client = _fake_groq_client("Here are three chairs you might like.")
    llm = GeneratorLLM(model_name="llama-3.1-8b-instant", client=client)
    messages = [{"role": "user", "content": "show me chairs"}]

    result = llm.complete(messages, temperature=0.5, max_tokens=100)

    assert result == "Here are three chairs you might like."
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "llama-3.1-8b-instant"
    assert kwargs["messages"] == messages
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 100


def test_complete_returns_empty_string_when_content_is_none():
    client = _fake_groq_client(None)
    llm = GeneratorLLM(model_name="llama-3.1-8b-instant", client=client)

    assert llm.complete([{"role": "user", "content": "hi"}]) == ""


# ---------------------------------------------------------------------------
# complete_structured — Pydantic-validated JSON-mode extraction
# ---------------------------------------------------------------------------


def test_complete_structured_requests_json_mode_and_validates_into_the_response_model():
    client = _fake_groq_client('{"color": "blue", "max_price": 100.0}')
    llm = GeneratorLLM(model_name="llama-3.1-8b-instant", client=client)

    result = llm.complete_structured([{"role": "user", "content": "blue chairs under $100"}], _Filters)

    assert isinstance(result, _Filters)
    assert result == _Filters(color="blue", max_price=100.0)
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.0  # extraction is deterministic, not creative


def test_complete_structured_injects_the_pydantic_schema_into_an_existing_system_message():
    client = _fake_groq_client('{"color": null, "max_price": null}')
    llm = GeneratorLLM(model_name="llama-3.1-8b-instant", client=client)
    messages = [
        {"role": "system", "content": "You are a shopping assistant."},
        {"role": "user", "content": "anything nice?"},
    ]

    llm.complete_structured(messages, _Filters)

    _, kwargs = client.chat.completions.create.call_args
    sent_system = kwargs["messages"][0]
    assert sent_system["role"] == "system"
    assert sent_system["content"].startswith("You are a shopping assistant.")
    assert "color" in sent_system["content"]  # the injected JSON-schema description
    assert len(kwargs["messages"]) == 2  # no extra message appended — augmented in place


def test_complete_structured_inserts_a_system_message_when_none_was_provided():
    client = _fake_groq_client('{"color": null, "max_price": null}')
    llm = GeneratorLLM(model_name="llama-3.1-8b-instant", client=client)
    messages = [{"role": "user", "content": "anything nice?"}]

    llm.complete_structured(messages, _Filters)

    _, kwargs = client.chat.completions.create.call_args
    assert len(kwargs["messages"]) == 2
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1] == messages[0]


def test_complete_structured_raises_on_a_response_that_violates_the_schema():
    client = _fake_groq_client('{"color": 12345}')  # color must be a string or null
    llm = GeneratorLLM(model_name="llama-3.1-8b-instant", client=client)

    with pytest.raises(
        Exception
    ):  # pydantic.ValidationError — a malformed response is a real failure, not silently swallowed
        llm.complete_structured([{"role": "user", "content": "..."}], _Filters)
