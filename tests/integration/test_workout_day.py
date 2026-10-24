"""Integration tests: the next workout day from a real plan and log in Postgres, and the
chat answering "what's my workout today?" with a FunctionModel standing in for the model.
"""

from datetime import date
from decimal import Decimal
from typing import Any, cast

import pytest
from pydantic import SecretStr
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.agents.workout_agent import build_agent
from workout_agent.settings import Settings
from workout_agent.tools.workout_day import planned_day, workout_days
from workout_agent.workflow.agent_chat import stream_agent_message
from workout_agent.workflow.events import Final, ToolEnd, ToolStart

AGENT = build_agent(Settings(openrouter_api_key=SecretStr("test")))
WEDNESDAY = date(2026, 10, 14)


@pytest.fixture
async def plan(conn):
    """Push A (bench, fly), Legs A (squat, plank), Arms (curl, fly); a short log."""
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [(2, "Barbell Bench Press"), (45, "Machine Fly"), (10, "Barbell Squat"),
         (631, "Plank"), (112, "Barbell Curl")],
    )
    await conn.execute("insert into routines values (1, 'PPL')")
    await conn.executemany(
        "insert into routine_days values ($1, 1, $2, $3, $4)",
        [(1, "Push A. Chest and shoulder", 1, True), (3, "Legs A", 3, True),
         (4, "Arms", 4, False)],
    )
    await conn.executemany(
        "insert into planned_exercises values ($1, $2, $3, $4, $5)",
        [(1, 1, 2, 0, [7, 8, 10]), (2, 1, 45, 1, [12, 10]),
         (3, 3, 10, 0, [10, 10]), (4, 3, 631, 1, [60]),
         (5, 4, 112, 0, [12, 12]), (6, 4, 45, 1, [12])],
    )
    await conn.executemany(
        "insert into exercise_logs values ($1, $2, $3)",
        [(1, 10, date(2026, 9, 23)), (2, 631, date(2026, 9, 23)),  # Legs A
         (3, 2, date(2026, 10, 5)), (4, 45, date(2026, 10, 5)),  # Push A
         (5, 112, date(2026, 10, 3)), (6, 45, date(2026, 10, 3))],  # Arms
    )
    await conn.executemany(
        "insert into sets values ($1, $2, $3, $3, $4)",
        [(1, 1, Decimal("40.00"), 10), (1, 2, Decimal("45.00"), 10),
         (3, 1, Decimal("30.00"), 7), (6, 1, Decimal("25.00"), 12)],
    )
    return conn


async def test_the_rotation_day_done_longest_ago_is_suggested(plan):
    result = await workout_days(plan, WEDNESDAY)
    assert result.suggested_day_id == 3  # Legs A; Arms is older but not in the rotation
    assert result.data_through == date(2026, 10, 5)
    assert [(d.name, d.last_done) for d in result.days] == [
        ("Push A. Chest and shoulder", date(2026, 10, 5)),
        ("Legs A", date(2026, 9, 23)),
        ("Arms", date(2026, 10, 3)),
    ]


async def test_a_day_lists_its_exercises_with_the_last_session(plan):
    push = await planned_day(plan, 1)
    assert [(x.exercise_name, x.target_reps, x.last_on) for x in push.exercises] == [
        ("Barbell Bench Press", [7, 8, 10], date(2026, 10, 5)),
        ("Machine Fly", [12, 10], date(2026, 10, 5)),  # its latest log, on any day
    ]
    assert [(s.weight_kg, s.reps) for s in push.exercises[0].last_sets] == [
        (Decimal("30.00"), 7)
    ]
    legs = await planned_day(plan, 3)
    assert [x.timed for x in legs.exercises] == [False, True]  # Plank is timed
    assert (await planned_day(plan, 99)).status == "no_rows"


async def test_no_plan_imported_is_no_rows(conn):
    assert (await workout_days(conn, WEDNESDAY)).status == "no_rows"


class TodaysWorkout:
    """workout_days, then ShowWorkoutDay with the day it suggested."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("workout_days_tool", {})])
        days = cast(dict[str, Any], returns[0].content)
        self.seen.append(str(days))
        return ModelResponse(parts=[ToolCallPart(
            "final_result_ShowWorkoutDay", {"day_id": days["suggested_day_id"]}
        )])


async def test_whats_my_workout_today(plan, checked_today):
    model = TodaysWorkout()
    with AGENT.override(model=FunctionModel(model)):
        events = [e async for e in stream_agent_message(plan, "t1", "what's my workout today?",
                                                        AGENT)]

    assert [type(e) for e in events[:-1]] == [ToolStart, ToolEnd, ToolStart, ToolEnd]
    day_id = events[0].args == {} and events[1].result["suggested_day_id"]
    assert events[2] == ToolStart(
        tool="planned_day",
        label="Fetching that day's exercises…",
        args={"day_id": day_id},
        call_id="planned_day",
    )
    assert events[3].result["day_name"] == "Legs A"
    assert events[3].result["exercises"][0]["exercise_name"] == "Barbell Squat"
    final = events[-1]
    assert isinstance(final, Final) and final.verified
    assert final.reply.startswith("Your next workout is probably Legs A:")
    assert "1. Barbell Squat: 2 sets of 10, 10 reps; last time (23 Sep 2026): " in final.reply
    assert "2. Plank: 1 set, timed; last done 23 Sep 2026, no set details" in final.reply
    assert "Barbell Squat" not in model.seen[0]  # the day's list never went to the model

    title = await plan.fetchval("select title from chats where chat_id = 't1'")
    assert title == "Legs A – workout"
