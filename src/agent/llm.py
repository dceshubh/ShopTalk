"""Generator LLM — Groq-hosted (not local Ollama; see `models.generator_llm` in
config.yaml and PROJECT_REPORT.md §5 for the memory-budget rationale behind that pivot).
Mirrors the `Encoder`/`load_encoder` pattern from `src.embeddings.encode`: one small
dataclass wrapping the provider client, loaded once and shared by every graph node that
needs to talk to the LLM (chat generation AND structured Pydantic extraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from groq import Groq
from pydantic import BaseModel

from src.common.config import load_config
from src.common.logging import get_logger
from src.common.secrets import get_env
from src.common.timer import Timer

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_STRUCTURED_SYSTEM_SUFFIX = (
    "\n\nRespond with a single JSON object matching this schema and nothing else "
    "(no prose, no markdown fences):\n{schema}"
)


@dataclass
class GeneratorLLM:
    """A Groq chat-completion client pinned to one model name."""

    model_name: str
    client: Groq

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Free-text chat completion — the conversational-response path (rephrasing
        retrieved products, asking an upsell follow-up)."""
        with Timer(f"llm.complete[{self.model_name}] n_messages={len(messages)}"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return response.choices[0].message.content or ""

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        """Structured-output completion: appends the Pydantic JSON schema to the system
        prompt, requests JSON mode, and validates the response into `response_model`.

        Pydantic-validated structured output (not regex on free-form text) per the
        filter-extraction exit gate — "returns valid Pydantic objects on 20 varied queries,
        no parse failures." `temperature=0.0` by default: extraction should be deterministic
        given the same input, not creative.
        """
        schema = response_model.model_json_schema()
        augmented = _augment_system_message(messages, schema)
        with Timer(f"llm.complete_structured[{self.model_name}] -> {response_model.__name__}"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=augmented,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content or "{}"
        return response_model.model_validate_json(content)


def _augment_system_message(messages: list[dict[str, str]], schema: dict) -> list[dict[str, str]]:
    """Append the target Pydantic schema to the (first) system message, or insert one if
    the caller didn't provide one. Keeps `complete_structured` callable with a plain
    user-only message list."""
    suffix = _STRUCTURED_SYSTEM_SUFFIX.format(schema=schema)
    augmented = list(messages)
    for i, message in enumerate(augmented):
        if message["role"] == "system":
            augmented[i] = {**message, "content": message["content"] + suffix}
            return augmented
    return [{"role": "system", "content": suffix.lstrip()}, *augmented]


def load_generator(model_name: str | None = None) -> GeneratorLLM:
    """Load the Groq-hosted generator LLM. Defaults to `models.generator_llm.primary`
    in config.yaml. Requires `GROQ_API_KEY` in `.env` (see `.env.example`)."""
    cfg = load_config()
    model_name = model_name or cfg["models"]["generator_llm"]["primary"]
    api_key = get_env("GROQ_API_KEY", required=True)
    logger.info("Loading Groq generator LLM %s", model_name)
    return GeneratorLLM(model_name=model_name, client=Groq(api_key=api_key))
