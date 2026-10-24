"""The once-a-day freshness check: the first chat of the day looks up the newest session
date and stores it outside the chat; later chats read the stored row and skip the lookup.
"""

import json
from datetime import date, timedelta

from pydantic import SecretStr
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import workout_agent.sessions.daily_checks as daily_checks
from workout_agent.agents.workout_agent import build_agent
from workout_agent.settings import Settings
from workout_agent.workflow.agent_chat import stream_agent_message
from workout_agent.workflow.events import Final, ToolEnd, ToolStart

AGENT = build_agent(Settings(openrouter_api_key=SecretStr("test")))
TODAY = date.today()
NEWEST = {"status": "ok", "latest_on": "2026-02-09"}  # the newest log in the `seeded` fixture


def help_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart("final_result_Help", {})])


async def say(conn, chat_id: str) -> list:
    with AGENT.override(model=FunctionModel(help_model)):
        return [e async for e in stream_agent_message(conn, chat_id, "hello", AGENT)]


def lookups(events: list) -> list:
    return [e for e in events if isinstance(e, ToolStart) and e.tool == "latest_record"]


async def rows(conn) -> list:
    return await conn.fetch("select checked_on, result from daily_checks order by checked_on")


async def never_queried(conn):
    raise AssertionError("the history was queried again")


async def test_the_first_message_of_the_day_looks_it_up_and_stores_it(conn, seeded):
    events = await say(conn, "t1")

    assert events[:2] == [
        ToolStart(tool="latest_record", label="Checking how recent your data is…"),
        ToolEnd(tool="latest_record", status="ok"),
    ]
    assert isinstance(events[-1], Final)
    [row] = await rows(conn)
    assert row["checked_on"] == TODAY
    assert json.loads(row["result"]) == NEWEST
    # The lookup is in the evidence saved with the turn.
    assert NEWEST in json.loads(await conn.fetchval("select evidence from chat_turns"))


async def test_later_chats_the_same_day_read_it_and_skip_the_lookup(conn, seeded, monkeypatch):
    await say(conn, "t1")
    monkeypatch.setattr(daily_checks, "latest_record", never_queried)

    other_chat = await say(conn, "t2")
    same_chat = await say(conn, "t1")

    assert lookups(other_chat) == [] and lookups(same_chat) == []
    assert isinstance(other_chat[-1], Final) and len(await rows(conn)) == 1
    # Read, not looked up: nothing of it in this chat's evidence.
    turns = await conn.fetch("select chat_id, evidence from chat_turns where chat_id = 't2'")
    assert NEWEST not in json.loads(turns[0]["evidence"])


async def test_the_next_day_looks_it_up_again(conn, seeded):
    tomorrow = TODAY + timedelta(days=1)
    await say(conn, "t1")

    stored, looked_up = await daily_checks.latest_record_today(conn, tomorrow)

    assert looked_up and stored == NEWEST
    assert [r["checked_on"] for r in await rows(conn)] == [TODAY, tomorrow]


async def test_no_data_is_stored_and_not_looked_up_again(conn, monkeypatch):
    events = await say(conn, "t1")

    assert ToolEnd(tool="latest_record", status="no_rows") in events
    [row] = await rows(conn)
    assert json.loads(row["result"]) == {"status": "no_rows", "latest_on": None}

    monkeypatch.setattr(daily_checks, "latest_record", never_queried)
    assert lookups(await say(conn, "t2")) == []


async def test_a_failed_lookup_does_not_block_the_answer(conn, seeded, monkeypatch):
    monkeypatch.setattr(daily_checks, "latest_record", never_queried)

    events = await say(conn, "t1")

    assert lookups(events) == []
    assert isinstance(events[-1], Final) and events[-1].verified
    assert await rows(conn) == []
