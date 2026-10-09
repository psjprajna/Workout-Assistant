"""Integration tests: the tool-calling agent, against a REAL language model.

These call OpenRouter, so they cost a little money and need OPENROUTER_API_KEY in .env.
They are marked `llm`: run them with `uv run pytest -m llm`, skip with `-m "not llm"`.
The model's wording changes every run, so the tests check facts and evidence
(did it call the tool? is every number from the data?), never exact sentences.

The agent lives in src/workout_agent/agents/workout_agent.py (Session 2). Until that
file exists the whole module is skipped, so the rest of the suite stays green.
"""

import re
from datetime import date
from decimal import Decimal

import pytest

from workout_agent.settings import Settings

# Skip (not fail) while the agent module has not been written yet.
agent_module = pytest.importorskip("workout_agent.agents.workout_agent")

# Applies the `llm` marker to every test in this file.
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        Settings().openrouter_api_key is None, reason="OPENROUTER_API_KEY is not set"
    ),
]


async def _seed(conn):
    """Two bench presses and one curl; only the curl has sets (30 kg x 10, 30 kg x 9)."""
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [(2, "Barbell Bench Press"), (8, "Barbell Incline Bench Press"), (112, "Barbell Curl")],
    )
    await conn.executemany(
        "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
        "values ($1, $2, $3)",
        [(10, 2, date(2026, 2, 8)), (11, 8, date(2026, 2, 9)), (12, 112, date(2026, 2, 10))],
    )
    await conn.executemany(
        "insert into sets (exercise_log_id, set_number, weight_kg, weight_raw, reps) "
        "values ($1, $2, $3, $4, $5)",
        [
            (12, 1, Decimal("30.00"), Decimal("30.00"), 10),
            (12, 2, Decimal("30.00"), Decimal("30.00"), 9),
        ],
    )


async def _ask(conn, question):
    """Run the agent once. Returns (reply text, evidence = the tool results it collected)."""
    deps = agent_module.AgentDeps(conn=conn, evidence=[])
    agent = agent_module.build_agent(Settings())
    result = await agent.run(question, deps=deps)
    return result.output, deps.evidence


async def test_answers_last_session_from_the_data(conn):
    await _seed(conn)
    reply, evidence = await _ask(conn, "What did I lift last time for barbell curl?")

    assert evidence, "the agent answered without calling any tool"
    assert "30" in reply  # the weight comes from the tool result


async def test_never_invents_numbers(conn):
    await _seed(conn)
    reply, _ = await _ask(conn, "What did I lift last time for barbell curl?")

    # Every number allowed in the reply: weights, reps, set numbers, date parts.
    allowed = {"30", "10", "9", "1", "2", "2026"}
    numbers = set(re.findall(r"\d+(?:\.\d+)?", reply))
    assert {n.rstrip("0").rstrip(".") if "." in n else n for n in numbers} <= allowed | {"02"}


async def test_unknown_exercise_is_reported_not_guessed(conn):
    await _seed(conn)
    reply, _ = await _ask(conn, "What did I lift last time for zzz?")

    assert not re.search(r"\d", reply), "no numbers should appear for an unknown exercise"


async def test_ambiguous_exercise_asks_which_one(conn):
    await _seed(conn)
    reply, _ = await _ask(conn, "What did I lift last time for bench press?")

    # Two bench presses match, so it should name them and ask, not pick one.
    assert "incline" in reply.lower()
    assert "?" in reply


async def test_picking_a_number_chooses_that_option(conn):
    await _seed(conn)
    agent = agent_module.build_agent(Settings())
    deps = agent_module.AgentDeps(conn=conn, evidence=[])
    first = await agent.run("last time I did bench press?", deps=deps)

    # The tool returned the options in this order; "2" must mean the second one.
    second_name = deps.evidence[0]["matches"][1]["name"]
    deps2 = agent_module.AgentDeps(conn=conn, evidence=[])
    second = await agent.run("2", deps=deps2, message_history=first.all_messages())

    assert second_name in second.output
    assert "1." in first.output  # the options were shown as a numbered list