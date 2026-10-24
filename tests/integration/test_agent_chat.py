"""Integration tests: the agent path saves each turn and gets it back after a restart,
streams tool progress as it runs, lets only one message per chat run at a time, and
writes replies from the model's typed choice and this turn's tool results only.

The real agent and tools run against Postgres, but a FunctionModel stands in for the
language model, so there are no network calls and the turns are predictable. The
FunctionModel answers the way the real one must: by calling an output tool
(final_result_ShowLastSession, ...), never with text.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import asyncpg
import httpx
import pytest
from pydantic import SecretStr
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.guardrails import ToolBlocked

from workout_agent.agents import workout_agent
from workout_agent.agents.workout_agent import build_agent
from workout_agent.api.app import create_app
from workout_agent.sessions import chat_store
from workout_agent.settings import Settings
from workout_agent.tools.get_last_session import LastSessionResult, get_last_session
from workout_agent.workflow import agent_chat
from workout_agent.workflow.agent_chat import (
    BUSY_REPLY,
    handle_agent_message,
    stream_agent_message,
)
from workout_agent.workflow.events import Error, Final, ToolEnd, ToolStart, chat_event_adapter
from workout_agent.workflow.replies import HELP_TEXT
from workout_agent.workflow.safety import SAFETY_REPLY
from workout_agent.workflow.simple_router import simple_route

AGENT = build_agent(Settings(openrouter_api_key=SecretStr("test")))

# What code writes for the seeded bench press (conftest.seeded).
BENCH_REPLY = "Last time you did Barbell Bench Press was 8 Feb 2026:\n- Set 1: 80.00 kg x 8"


def choose(output: str, **args: Any) -> ModelResponse:
    """The model's final answer: a call to one of the output tools."""
    return ModelResponse(parts=[ToolCallPart(f"final_result_{output}", args)])


def call(tool: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args)])


class FakeModel:
    """Each turn: find bench press, get_last_session for it (id 2), choose ShowLastSession.

    Remembers the messages it was given at the start of every turn.
    """

    def __init__(self) -> None:
        self.seen: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        returns = [p for p in last.parts if isinstance(p, ToolReturnPart)]
        if returns and returns[0].tool_name == "find_exercise_tool":
            return call("get_last_session_tool", exercise_id=2)
        if returns:
            return choose("ShowLastSession", exercise_id=2)
        self.seen.append(list(messages))
        return call("find_exercise_tool", query="barbell bench press")


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
            assert await _chat(c, "t1", "last time bench press?") == BENCH_REPLY
        async with _api(test_database_url) as c:  # restart
            await _chat(c, "t1", "and again?")

    assert model.seen[0][:-1] == []  # turn 1: no history, just the question
    history = model.seen[1]
    sessions = [r for r in _tool_returns(history) if r.tool_name == "get_last_session_tool"]
    session = cast(dict[str, Any], sessions[0].content)
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
    """find_exercise, then get_last_session for id 2, then choose ShowLastSession."""

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return call("find_exercise_tool", query="barbell bench press")
        if returns[0].tool_name == "find_exercise_tool":
            return call("get_last_session_tool", exercise_id=2)
        return choose("ShowLastSession", exercise_id=2)


async def _turns(conn) -> int:
    return await conn.fetchval("select count(*) from chat_turns")


async def test_stream_reports_each_tool_then_the_reply(conn, seeded, checked_today):
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
        Final(reply=BENCH_REPLY, verified=True, checked=2),
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
    assert reply == BENCH_REPLY


async def test_a_message_while_the_chat_is_answering_is_busy_at_once(
    conn, seeded, test_database_url
):
    calls = []

    def model(messages, info):
        calls.append(messages)
        return choose("Help")

    other = await asyncpg.connect(test_database_url)
    try:
        # Another request is answering in chat t1: it holds the chat's lock.
        assert await chat_store.try_lock_chat(other, "t1")
        with AGENT.override(model=FunctionModel(model)):
            events = [e async for e in stream_agent_message(conn, "t1", "bench?", AGENT)]
            # Other chats are not affected.
            assert await handle_agent_message(conn, "t2", "bench?", AGENT) == HELP_TEXT
    finally:
        await other.close()

    assert events == [Error(message=BUSY_REPLY)]
    assert len(calls) == 1  # only chat t2 reached the model
    assert await conn.fetchval("select count(*) from chat_turns where chat_id = 't1'") == 0


async def test_chat_stream_endpoint_sends_sse_events(seeded, checked_today, test_database_url):
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
    assert events[-1] == Final(reply=BENCH_REPLY, verified=True, checked=2)


async def test_chat_stream_endpoint_turns_a_failure_into_an_error_event(
    seeded, checked_today, test_database_url
):
    def broken(messages, info):
        raise RuntimeError("model down")

    with AGENT.override(model=FunctionModel(broken)):
        async with _api(test_database_url) as c:
            r = await c.post("/chat/stream", json={"chat_id": "t1", "message": "bench?"})

    data = [line for line in r.text.splitlines() if line.startswith("data: ")]
    assert len(data) == 1
    assert isinstance(chat_event_adapter.validate_json(data[0].removeprefix("data: ")), Error)


# --- Typed answers: code writes the reply, only from this turn's tool results --------


async def test_an_answer_about_an_unfetched_id_is_sent_back_to_the_model(
    conn, seeded, checked_today
):
    retries: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = messages[-1].parts
        retry = [p for p in parts if isinstance(p, RetryPromptPart)]
        if retry:  # told to fetch first: does so
            retries.append(str(retry[0].content))
            return call("find_exercise_tool", query="barbell bench press")
        returns = [p for p in parts if isinstance(p, ToolReturnPart)]
        if returns and returns[0].tool_name == "find_exercise_tool":
            return call("get_last_session_tool", exercise_id=2)
        return choose("ShowLastSession", exercise_id=2)  # first: answers without fetching

    with AGENT.override(model=FunctionModel(model)):
        events = [e async for e in stream_agent_message(conn, "t1", "bench?", AGENT)]

    assert "get_last_session was not called for exercise_id 2" in retries[0]
    assert events[-1] == Final(reply=BENCH_REPLY, verified=True, checked=2)


async def test_a_model_that_keeps_citing_unfetched_ids_saves_nothing(conn, seeded):
    def model(messages, info):
        return choose("ShowLastSession", exercise_id=8)  # never fetched

    with AGENT.override(model=FunctionModel(model)), pytest.raises(UnexpectedModelBehavior):
        await handle_agent_message(conn, "t1", "bench?", AGENT)
    assert await _turns(conn) == 0


class AskThenPick:
    """Turn 1: find_exercise -> AskWhich with its ids. Turn 2 ("2"): the 2nd id I offered."""

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if returns and returns[0].tool_name == "find_exercise_tool":
            matches = cast(dict[str, Any], returns[0].content)["matches"]
            return choose("AskWhich", option_ids=[m["exercise_id"] for m in matches])
        if returns and returns[0].tool_name == "get_last_session_tool":
            exercise_id = cast(dict[str, Any], returns[0].content)["exercise_id"]
            return choose("ShowLastSession", exercise_id=exercise_id)
        user_text = str(messages[-1].parts[-1].content)  # pyright: ignore[reportAttributeAccessIssue]
        if user_text.isdigit():
            asked = [
                p.args_as_dict()["option_ids"]
                for m in messages if isinstance(m, ModelResponse)
                for p in m.parts
                if isinstance(p, ToolCallPart) and p.tool_name == "final_result_AskWhich"
            ][-1]
            return call("get_last_session_tool", exercise_id=asked[int(user_text) - 1])
        return call("find_exercise_tool", query="bench press")


async def test_ask_which_then_a_number_shows_that_option(conn, seeded):
    with AGENT.override(model=FunctionModel(AskThenPick())):
        question = await handle_agent_message(conn, "t1", "last bench press?", AGENT)
        answer = await handle_agent_message(conn, "t1", "2", AGENT)

    assert question.startswith("I found more than one match for 'bench press'.")
    second = question.splitlines()[-1].removeprefix("2. ")
    assert answer.startswith(f"Last time you did {second} was")
    assert await _turns(conn) == 2


# --- Guardrails: the safety screen, tool calls and tool results ----------------------


async def test_a_safety_message_is_refused_without_calling_the_model(
    conn, seeded, checked_today
):
    def never(messages, info):
        raise AssertionError("the model must not see a refused message")

    with AGENT.override(model=FunctionModel(never)):
        events = [
            e async for e in stream_agent_message(conn, "t1", "my knee hurts, squat?", AGENT)
        ]
    assert events == [Final(reply=SAFETY_REPLY, verified=True, checked=0)]

    # Saved, so the chat shows it after a reload, but with no model messages.
    turns = await conn.fetch("select user_text, reply_text, messages from chat_turns")
    assert [(t["user_text"], t["reply_text"]) for t in turns] == [
        ("my knee hurts, squat?", SAFETY_REPLY)
    ]
    model = FakeModel()
    with AGENT.override(model=FunctionModel(model)):
        assert await handle_agent_message(conn, "t1", "bench?", AGENT) == BENCH_REPLY
    assert model.seen[0][:-1] == []  # the next turn's model never sees the refused text


async def test_a_made_up_id_is_sent_back_until_find_exercise_is_called(conn, seeded):
    retries: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = messages[-1].parts
        retry = [p for p in parts if isinstance(p, RetryPromptPart)]
        if retry:
            retries.append(str(retry[0].content))
            return call("find_exercise_tool", query="barbell bench press")
        returns = [p for p in parts if isinstance(p, ToolReturnPart)]
        if returns and returns[0].tool_name == "find_exercise_tool":
            return call("get_last_session_tool", exercise_id=2)
        if returns:
            return choose("ShowLastSession", exercise_id=2)
        return call("get_last_session_tool", exercise_id=2)  # guesses the id

    with AGENT.override(model=FunctionModel(model)):
        reply = await handle_agent_message(conn, "t1", "bench?", AGENT)

    assert retries == ["exercise_id 2 did not come from find_exercise; call find_exercise first"]
    assert reply == BENCH_REPLY


@dataclass
class _Leaky(LastSessionResult):
    birth_date: str = "1990-01-01"  # a field no tool may ever return


async def test_a_tool_result_outside_the_contract_stops_the_run(conn, seeded, monkeypatch):
    async def leaky(conn, exercise_id):
        return _Leaky(**vars(await get_last_session(conn, exercise_id)))

    monkeypatch.setattr(workout_agent, "get_last_session", leaky)
    with AGENT.override(model=FunctionModel(FindThenFetch())), pytest.raises(ToolBlocked):
        await handle_agent_message(conn, "t1", "bench?", AGENT)
    assert await _turns(conn) == 0
