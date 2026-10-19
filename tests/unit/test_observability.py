"""Unit tests: tracing stays off without keys; a turn's spans nest under it, with the chat id.

Spans go to an in-memory exporter, so no Langfuse server is needed.
"""

import asyncio
import uuid

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from workout_agent import observability
from workout_agent.observability import TurnTrace, setup_tracing
from workout_agent.settings import Settings


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4().hex}",  # Langfuse keeps one client per key
        secret_key="sk-test",
        base_url="http://127.0.0.1:9",  # nothing listens; scores are dropped
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    monkeypatch.setattr(observability, "_client", client)
    yield lambda: (client.flush(), exporter.get_finished_spans())[1]
    client.shutdown()


def test_no_keys_means_tracing_off(monkeypatch):
    monkeypatch.setattr(observability, "_client", None)
    assert setup_tracing(Settings(_env_file=None)) is False
    TurnTrace("chat-1", "hi").finish("help", "Hello", True)  # does nothing, raises nothing


async def test_work_in_the_turn_context_nests_under_the_turn(spans):
    turn = TurnTrace("chat-1", "last bench press?")

    async def agent_run():
        assert observability._client is not None
        with observability._client.start_as_current_observation(name="agent run"):
            await asyncio.sleep(0)

    await asyncio.create_task(agent_run(), context=turn.context)
    turn.finish("ShowLastSession", "Last time you did ...", True)

    by_name = {s.name: s for s in spans()}
    root, child = by_name["chat-turn"], by_name["agent run"]
    assert child.parent is not None and child.parent.span_id == root.context.span_id
    for span in (root, child):
        assert span.attributes is not None
        assert span.attributes["session.id"] == "chat-1"


async def test_the_callers_context_is_left_alone(spans):
    turn = TurnTrace("chat-1", "hi")
    assert observability._client is not None
    assert observability._client.get_current_trace_id() is None  # the span is not ours
    turn.finish("help")
    turn.finish("help")  # a second finish is ignored
    assert [s.name for s in spans()] == ["chat-turn"]
