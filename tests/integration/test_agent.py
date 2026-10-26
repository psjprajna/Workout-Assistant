"""Integration tests: the tool-calling agent, against a REAL language model.

These call OpenRouter, so they cost a little money and need OPENROUTER_API_KEY in .env.
They are marked `llm`: run them with `uv run pytest -m llm`, skip with `-m "not llm"`.
The model answers with a typed choice and code writes the reply from the tool results
(replies.render), so the tests render the reply the same way the chat does, then check
facts and evidence (did it call the tool? is every number from the data?).

The agent lives in src/workout_agent/agents/workout_agent.py (Session 2). Until that
file exists the whole module is skipped, so the rest of the suite stays green.
"""

import re
from datetime import date
from decimal import Decimal
from functools import partial

import pytest
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.index import index_exercises
from workout_agent.embeddings.openrouter import embed
from workout_agent.embeddings.search import search_exercises
from workout_agent.settings import Settings
from workout_agent.workflow.agent_chat import handle_agent_message
from workout_agent.workflow.replies import render

SEARCH_COLLECTION = "exercises_agent_test"

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
    return render(result.output, deps.evidence), deps.evidence


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
    # Through the chat path: code, not the model, turns "2" into the second option. The
    # seeded ids (2 and 8) are chosen so "2" would be the wrong answer if read as an id.
    first = await handle_agent_message(conn, "pick", "last time I did bench press?", agent)
    options = {
        line.split(". ", 1)[0]: line.split(". ", 1)[1]
        for line in first.splitlines()
        if re.match(r"\d+\. ", line)
    }
    assert set(options) >= {"1", "2"}, f"options shown as a numbered list: {first!r}"

    second = await handle_agent_message(conn, "pick", "2", agent)

    assert options["2"] in second


async def _real_embedder(texts: list[str]) -> list[list[float]]:
    return await embed(Settings(), texts)


@pytest.fixture
async def search(conn):
    """The _seed exercises indexed with the real embedder; yields a bound search_exercises."""
    await _seed(conn)
    qdrant = AsyncQdrantClient(url=Settings().qdrant_url)
    await qdrant.delete_collection(SEARCH_COLLECTION)
    try:
        await index_exercises(conn, qdrant, _real_embedder, SEARCH_COLLECTION)
        yield partial(
            search_exercises, qdrant, _real_embedder, collection=SEARCH_COLLECTION
        )
    finally:
        await qdrant.delete_collection(SEARCH_COLLECTION)
        await qdrant.close()


async def test_meaning_match_offers_bench_press(conn, search):
    deps = agent_module.AgentDeps(conn=conn, evidence=[], search=search)
    agent = agent_module.build_agent(Settings())
    result = await agent.run("what did I do last time for chest press?", deps=deps)

    # No name contains "chest", so the match comes from meaning, offered as a question.
    first = deps.evidence[0]
    assert first["status"] == "ambiguous"
    assert 2 in [m["exercise_id"] for m in first["matches"]]
    reply = render(result.output, deps.evidence)
    assert "Barbell Bench Press" in reply
    assert "?" in reply
    # It offered the options; it did not pick one and answer.
    assert all("sets" not in e for e in deps.evidence)
