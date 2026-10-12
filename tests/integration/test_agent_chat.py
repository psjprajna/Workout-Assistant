"""Integration tests: the agent path saves each turn and gets it back after a restart.

The real agent and tools run against Postgres, but a FunctionModel stands in for the
language model, so there are no network calls and the turns are predictable.
"""

from contextlib import asynccontextmanager
from typing import Any, cast

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
from workout_agent.workflow.agent_chat import BUSY_REPLY, handle_agent_message
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
