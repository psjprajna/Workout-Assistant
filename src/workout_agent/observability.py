"""Tracing into Langfuse (docs/amendments.md §25): one trace per chat turn, grouped by chat.

setup_tracing turns it on once per process: the Langfuse client exports OpenTelemetry
spans, and Agent.instrument_all() makes Pydantic AI emit one span per agent run, model
request and tool call (arguments, results, token counts). Without both keys in Settings
tracing stays off, and TurnTrace does nothing, so tests never need a Langfuse server.

TurnTrace is the span around one turn. stream_agent_message is an async generator, so
the span cannot be "current" in its own frame: the caller's context changes between
yields. Instead the span lives in a copied context, the agent run is started in that
context (asyncio.create_task(..., context=turn.context)), and the span is closed there.
"""

import contextvars
from typing import Any

from langfuse import Langfuse, LangfuseSpan, propagate_attributes
from langfuse.api import ScoreDataType
from pydantic_ai import Agent

from workout_agent.settings import Settings

_client: Langfuse | None = None


def setup_tracing(settings: Settings) -> bool:
    """Start exporting traces to Langfuse. False (and nothing changes) without keys."""
    global _client
    if _client is not None:
        return True
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return False
    _client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_host,
    )
    Agent.instrument_all()
    return True


def langfuse_client() -> Langfuse | None:
    """The client setup_tracing made (for datasets and experiments), or None."""
    return _client


def flush_tracing() -> None:
    """Send buffered spans now (before a script exits or the server stops)."""
    if _client is not None:
        _client.flush()


class TurnTrace:
    """The span for one chat turn: input is the user's text, session is the chat id."""

    def __init__(self, chat_id: str, text: str) -> None:
        self.context = contextvars.copy_context()
        self._span: LangfuseSpan | None = None
        self._managers: list[Any] = []
        if _client is not None:
            self.context.run(self._start, _client, chat_id, text)

    def _start(self, client: Langfuse, chat_id: str, text: str) -> None:
        span_cm = client.start_as_current_observation(name="chat-turn", input=text)
        self._span = span_cm.__enter__()
        # After the span is current, so the session id lands on it and on every child.
        attrs_cm = propagate_attributes(session_id=chat_id)
        attrs_cm.__enter__()
        self._managers = [attrs_cm, span_cm]

    def finish(self, outcome: str, reply: str | None = None, verified: bool | None = None) -> None:
        """Record how the turn ended and close the span. Call once, after the run is done."""
        if self._span is None:
            return
        span, self._span = self._span, None
        span.update(output=reply, metadata={"outcome": outcome, "verified": verified})
        if verified is not None:
            span.score(
                name="verified", value=1.0 if verified else 0.0, data_type=ScoreDataType.BOOLEAN
            )
        self.context.run(self._close)

    def tool_span(self, name: str, input: Any, output: Any) -> None:
        """Record a tool that code (not the model) called, as a child of this turn's span."""
        if self._span is None:
            return
        self.context.run(
            lambda: self._span.start_observation(  # pyright: ignore[reportOptionalMemberAccess]
                name=name, as_type="tool", input=input, output=output
            ).end()
        )

    def _close(self) -> None:
        for manager in self._managers:
            manager.__exit__(None, None, None)
        self._managers = []
