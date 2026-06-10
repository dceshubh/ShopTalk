"""Unit + integration tests for src.agent.memory.

`ConversationBuffer` (short-term/RAM) is pure and tested directly. `PersistentMemory`
(Redis-backed) is tested against a REAL local Redis (started via `brew services start
redis` — no Docker needed, see .env.example) using a dedicated `db=15` so it can never
collide with dev data on `db=0`; each test flushes its own keys. This is deliberate: the
Phase-5 exit gate is "a persisted pref survives a session restart," and a faked Redis
client can't prove that a `model_validate_json(model_dump_json(...))` round-trip actually
works against the real wire format.
"""

from __future__ import annotations

import pytest
import redis

from src.agent.memory import ConversationBuffer, PersistentMemory, UserPreferences, load_persistent_memory

TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture
def memory():
    client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not running locally — `brew services start redis` to enable this test")
    client.flushdb()
    yield PersistentMemory(redis_client=client)
    client.flushdb()


# ---------------------------------------------------------------------------
# ConversationBuffer — pure, in-RAM
# ---------------------------------------------------------------------------


def test_conversation_buffer_records_turns_in_order():
    buffer = ConversationBuffer()
    buffer.add_user("show me chairs")
    buffer.add_assistant("Here are three chairs.")
    buffer.add_user("cheaper ones?")

    assert buffer.as_messages() == [
        {"role": "user", "content": "show me chairs"},
        {"role": "assistant", "content": "Here are three chairs."},
        {"role": "user", "content": "cheaper ones?"},
    ]


def test_conversation_buffer_caps_at_max_turns_keeping_the_most_recent():
    buffer = ConversationBuffer(max_turns=2)
    for i in range(5):
        buffer.add_user(f"message {i}")
        buffer.add_assistant(f"reply {i}")

    messages = buffer.as_messages()
    assert len(messages) == 4  # 2 turns x (user + assistant)
    assert messages[0] == {"role": "user", "content": "message 3"}
    assert messages[-1] == {"role": "assistant", "content": "reply 4"}


def test_conversation_buffer_as_messages_returns_a_copy_not_the_live_list():
    buffer = ConversationBuffer()
    buffer.add_user("hello")
    snapshot = buffer.as_messages()
    buffer.add_assistant("hi there")

    assert snapshot == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# PersistentMemory — Redis-backed, real local Redis
# ---------------------------------------------------------------------------


def test_load_returns_an_empty_profile_for_a_never_seen_user(memory):
    prefs = memory.load("brand-new-user")

    assert prefs == UserPreferences()


def test_save_then_load_round_trips_through_real_redis(memory):
    original = UserPreferences(
        typical_recipient="my 5-year-old son", budget_ceiling=50.0, preferred_colors=["blue", "green"]
    )

    memory.save("user-1", original)
    reloaded = memory.load("user-1")

    assert reloaded == original


def test_persisted_pref_survives_a_fresh_client_connection(memory):
    """The actual exit-gate scenario: write -> "restart" (a brand-new client/connection,
    standing in for a process restart) -> recall."""
    memory.save("user-2", UserPreferences(typical_recipient="my niece", preferred_size="M"))

    fresh_client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    fresh_memory = PersistentMemory(redis_client=fresh_client)

    assert fresh_memory.load("user-2") == UserPreferences(typical_recipient="my niece", preferred_size="M")


def test_merge_layers_new_fields_onto_the_existing_profile_without_erasing_them(memory):
    memory.save("user-3", UserPreferences(typical_recipient="my daughter", budget_ceiling=100.0))

    # This turn only mentions color — must not wipe out recipient/budget learned earlier.
    merged = memory.merge("user-3", UserPreferences(preferred_colors=["pink"]))

    assert merged == UserPreferences(
        typical_recipient="my daughter",
        budget_ceiling=100.0,
        preferred_colors=["pink"],
    )
    assert memory.load("user-3") == merged  # merge persists, not just returns


def test_merge_overwrites_a_field_when_the_new_turn_provides_a_newer_value(memory):
    memory.save("user-4", UserPreferences(budget_ceiling=50.0))

    merged = memory.merge("user-4", UserPreferences(budget_ceiling=75.0))

    assert merged.budget_ceiling == 75.0


# ---------------------------------------------------------------------------
# load_persistent_memory — URL resolution priority (explicit arg > REDIS_URL env > config)
#
# This priority is what lets the SAME container image run as local dev (Redis on
# localhost), inside docker-compose (Redis reached by service name), and on AWS — see
# the function's docstring for the full "one image, three runtime addresses" rationale.
# ---------------------------------------------------------------------------


def test_load_persistent_memory_prefers_an_explicit_url_over_everything(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://from-env:6379/0")

    memory = load_persistent_memory("redis://explicit:6379/0")

    assert memory.redis_client.connection_pool.connection_kwargs["host"] == "explicit"


def test_load_persistent_memory_falls_back_to_the_redis_url_env_var(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://from-env:6379/0")

    memory = load_persistent_memory()

    assert memory.redis_client.connection_pool.connection_kwargs["host"] == "from-env"


def test_load_persistent_memory_falls_back_to_config_when_nothing_else_is_set(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    memory = load_persistent_memory()

    # configs/config.yaml: agent.memory.redis_url -> "redis://localhost:6379/0"
    assert memory.redis_client.connection_pool.connection_kwargs["host"] == "localhost"
