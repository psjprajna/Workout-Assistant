"""Integration tests: the agent path saves each turn and gets it back after a restart,
streams tool progress as it runs, and lets only one message per chat run at a time.

The real agent and tools run against Postgres, but a FunctionModel stands in for the
language model, so there are no network calls and the turns are predictable.
"""

from contextlib import asynccontextmanager
from typing import Any, cast

import asyncpg
import httpx
import pytest
from pydantic import SecretStr
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.agents.workout_agent import build_agent
from workout_agent.api.app import create_app
from workout_agent.sessions import chat_store
from workout_agent.settings import Settings
from workout_agent.workflow import agent_chat
from workout_agent.workflow.agent_chat import (
    BUSY_REPLY,
    handle_agent_message,
    stream_agent_message,
)
from workout_agent.workflow.events import Error, Final, ToolEnd, ToolStart, chat_event_adapter
from workout_agent.workflow.replies import UNVERIFIED_REPLY
from workout_agent.workflow.simple_router import simple_route

AGENT = build_agent(Settings(openrouter_api_key=SecretStr("test")))


class FakeModel:
    """Each turn: get_last_session for bench press (id 2), then answer from its result.

    Remembers the messages it was given at the start of every turn.
    """

    def __init__(self) -> None:
        self.seen: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        returns = [p for p in last.parts if isinstance(p, ToolReturnPart)]
        if returns:
            sets = cast(dict[str, Any], returns[0].content)["sets"]
            return ModelResponse(parts=[TextPart(f"{sets[0]['weight_kg']} kg")])
        self.seen.append(list(messages))
        return ModelResponse(
            parts=[ToolCallPart("get_last_session_tool", {"exercise_id": 2}, f"c{len(self.seen)}")]
        )


@asynccontextmanager
async def _api(database_url):
    """A fresh app, as after a restart: nothing survives except Postgres."""
    app = create_app(route=simple_route, database_url=database_url, agent=AGENT)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def _chat(client, chat_id, message):
    r = await client.post("/chat", json={"chat_id": chat_id, "message": message})
    assert r.status_code == 200
    return r.json()["reply"]


def _tool_returns(messages) -> list[ToolReturnPart]:
    return [
        p for m in messages if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, ToolReturnPart)
    ]


async def test_second_turn_sees_the_first_after_a_restart(seeded, test_database_url):
    model = FakeModel()
    with AGENT.override(model=FunctionModel(model)):
        async with _api(test_database_url) as c:
            assert await _chat(c, "t1", "last time bench press?") == "80.00 kg"
        async with _api(test_database_url) as c:  # restart
            await _chat(c, "t1", "and again?")

    assert model.seen[0][:-1] == []  # turn 1: no history, just the question
    history = model.seen[1]
    returns = _tool_returns(history)
    assert returns[0].tool_name == "get_last_session_tool"
    session = cast(dict[str, Any], returns[0].content)
    assert session["sets"][0]["weight_kg"] == "80.00"  # recent turn kept whole

    chats = await _list(test_database_url)
    assert [(ch["chat_id"], ch["title"]) for ch in chats] == [
        ("t1", "Barbell Bench Press – last session")
    ]


async def _list(database_url):
    async with _api(database_url) as c:
        return (await c.get("/chats")).json()


async def test_other_chats_get_no_history(seeded, test_database_url):
    model = FakeModel()
    with AGENT.override(model=FunctionModel(model)):
        async with _api(test_database_url) as c:
            await _chat(c, "a", "bench?")
            await _chat(c, "b", "bench?")
    assert len(model.seen[1]) == 1  # only chat b's own question


async def test_failed_run_saves_nothing(conn, seeded):
    def broken(messages, info):
        raise RuntimeError("model down")

    with AGENT.override(model=FunctionModel(broken)), pytest.raises(RuntimeError):
        await handle_agent_message(conn, "t1", "bench?", AGENT)
    assert await conn.fetchval("select count(*) from chats") == 0


async def test_a_turn_saved_meanwhile_gives_the_busy_reply(conn, seeded, monkeypatch):
    with AGENT.override(model=FunctionModel(FakeModel())):
        await handle_agent_message(conn, "t1", "first", AGENT)

        # Another request for the same chat saves turn 2 while this one is running.
        async def load_then_race(conn, chat_id, last_turns):
            loaded = await chat_store.load_history(conn, chat_id, last_turns)
            await chat_store.save_turn(conn, chat_id, 2, "other", "other", [], [], "x", True)
            return loaded

        monkeypatch.setattr(agent_chat, "load_history", load_then_race)
        reply = await handle_agent_message(conn, "t1", "second", AGENT)

    assert reply == BUSY_REPLY
    turns = await conn.fetch("select user_text from chat_turns order by turn_no")
    assert [t["user_text"] for t in turns] == ["first", "other"]


# --- Streaming: tool progress first, then the reply ----------------------------------


class FindThenFetch:
    """find_exercise, then get_last_session for id 2, then answer from its result."""

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            call = ToolCallPart("find_exercise_tool", {"query": "barbell bench press"}, "f")
            return ModelResponse(parts=[call])
        if returns[0].tool_name == "find_exercise_tool":
            call = ToolCallPart("get_last_session_tool", {"exercise_id": 2}, "g")
            return ModelResponse(parts=[call])
        sets = cast(dict[str, Any], returns[0].content)["sets"]
        return ModelResponse(parts=[TextPart(f"{sets[0]['weight_kg']} kg")])


async def _turns(conn) -> int:
    return await conn.fetchval("select count(*) from chat_turns")


async def test_stream_reports_each_tool_then_the_reply(conn, seeded):
    with AGENT.override(model=FunctionModel(FindThenFetch())):
        events = [e async for e in stream_agent_message(conn, "t1", "bench?", AGENT)]

    assert events == [
        ToolStart(
            tool="find_exercise_tool",
            label="Searching your exercises for 'barbell bench press'…",
        ),
        ToolEnd(tool="find_exercise_tool", status="ok"),
        ToolStart(tool="get_last_session_tool", label="Fetching your last session…"),
        ToolEnd(tool="get_last_session_tool", status="ok"),
        Final(reply="80.00 kg", verified=True, checked=2),
    ]
    assert await _turns(conn) == 1


async def test_a_stream_closed_early_saves_nothing_and_frees_the_chat(conn, seeded):
    with AGENT.override(model=FunctionModel(FindThenFetch())):
        stream = stream_agent_message(conn, "t1", "bench?", AGENT)
        first = await anext(stream)
        await stream.aclose()  # the client went away after the first event

        assert isinstance(first, ToolStart)
        assert await _turns(conn) == 0
        reply = await handle_agent_message(conn, "t1", "bench?", AGENT)  # not busy
    assert reply == "80.00 kg"


async def test_a_message_while_the_chat_is_answering_is_busy_at_once(
    conn, seeded, test_database_url
):
    calls = []

    def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("hi")])

    other = await asyncpg.connect(test_database_url)
    try:
        # Another request is answering in chat t1: it holds the chat's lock.
        assert await chat_store.try_lock_chat(other, "t1")
        with AGENT.override(model=FunctionModel(model)):
            events = [e async for e in stream_agent_message(conn, "t1", "bench?", AGENT)]
            # Other chats are not affected.
            assert await handle_agent_message(conn, "t2", "bench?", AGENT) == "hi"
    finally:
        await other.close()

    assert events == [Error(message=BUSY_REPLY)]
    assert len(calls) == 1  # only chat t2 reached the model
    assert await conn.fetchval("select count(*) from chat_turns where chat_id = 't1'") == 0


async def test_chat_stream_endpoint_sends_sse_events(seeded, test_database_url):
    with AGENT.override(model=FunctionModel(FindThenFetch())):
        async with _api(test_database_url) as c:
            async with c.stream(
                "POST", "/chat/stream", json={"chat_id": "t1", "message": "bench?"}
            ) as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                lines = [line async for line in r.aiter_lines() if line.startswith("data: ")]

    events = [chat_event_adapter.validate_json(line.removeprefix("data: ")) for line in lines]
    assert [e.kind for e in events] == ["tool_start", "tool_end", "tool_start", "tool_end", "final"]
    assert events[-1] == Final(reply="80.00 kg", verified=True, checked=2)


async def test_chat_stream_endpoint_turns_a_failure_into_an_error_event(
    seeded, test_database_url
):
    def broken(messages, info):
        raise RuntimeError("model down")

    with AGENT.override(model=FunctionModel(broken)):
        async with _api(test_database_url) as c:
            r = await c.post("/chat/stream", json={"chat_id": "t1", "message": "bench?"})

    data = [line for line in r.text.splitlines() if line.startswith("data: ")]
    assert len(data) == 1
    assert isinstance(chat_event_adapter.validate_json(data[0].removeprefix("data: ")), Error)


# --- Verification: no number the tools did not return ---------------------------------


def _answers(text: str):
    """get_last_session for id 2, then reply with `text` whatever the tool said."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(text)])
        return ModelResponse(parts=[ToolCallPart("get_last_session_tool", {"exercise_id": 2})])

    return model


async def test_an_invented_number_is_refused_and_not_saved(conn, seeded):
    with AGENT.override(model=FunctionModel(_answers("You lifted 85.00 kg x 8."))):
        events = [e async for e in stream_agent_message(conn, "t1", "bench?", AGENT)]

    assert events[-1] == Final(reply=UNVERIFIED_REPLY, verified=False, checked=1)
    assert await _turns(conn) == 0


async def test_a_follow_up_may_use_numbers_from_earlier_tool_results(conn, seeded):
    def no_tools(messages, info):
        return ModelResponse(parts=[TextPart("That was 8 reps at 80 kg.")])

    with AGENT.override(model=FunctionModel(FakeModel())):
        await handle_agent_message(conn, "t1", "bench?", AGENT)
    with AGENT.override(model=FunctionModel(no_tools)):  # answers from history alone
        events = [e async for e in stream_agent_message(conn, "t1", "reps?", AGENT)]

    assert events == [Final(reply="That was 8 reps at 80 kg.", verified=True, checked=1)]
    assert await _turns(conn) == 2
