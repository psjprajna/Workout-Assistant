"""The red team against the agent (docs/threat-model.md, src/workout_agent/evals/redteam.py).

- The scripted tests run everywhere: a FunctionModel that is fooled by a hostile name must be
  scored as an attack that worked, and a faithful one must not, so the scorer is itself tested.
- test_the_red_team_cannot_break_the_agent (marker `eval`) runs every case on the real model,
  twice, and fails on any success: the gate has no slack and no baseline.
"""

from dataclasses import asdict
from typing import Any

import asyncpg
import pytest
from pydantic import SecretStr
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.agents.workout_agent import build_agent
from workout_agent.evals.agent import FIND, LAST, run_case
from workout_agent.evals.gate import check_redteam
from workout_agent.evals.redteam import (
    CASES,
    RedCase,
    injection_succeeded,
    seed_redteam,
    summarize_redteam,
)
from workout_agent.evals.synthetic import prepare_database, seed
from workout_agent.settings import Settings

REPEATS = 2


async def play(url: str, agent: Any, name: str, case: RedCase) -> list[dict[str, Any]]:
    """One case on the seeded history with its hostile names, in a rolled-back transaction."""
    conn = await asyncpg.connect(url)
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await seed(conn)
            await seed_redteam(conn, case.poison)
            return [asdict(r) for r in await run_case(conn, agent, name, case.turns)]
        finally:
            await tx.rollback()
    finally:
        await conn.close()


def curl_model(obeys_the_name: bool):
    """Barbell curl, find then fetch then show; or, if fooled by the name, Help after the find."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart(FIND, {"query": "barbell curl"})])
        if returns[0].tool_name == FIND:
            if obeys_the_name:
                return ModelResponse(parts=[ToolCallPart("final_result_Help", {})])
            return ModelResponse(parts=[ToolCallPart(LAST, {"exercise_id": 20})])
        return ModelResponse(
            parts=[ToolCallPart("final_result_ShowLastSession", {"exercise_id": 20})]
        )

    return model


@pytest.mark.parametrize(("obeys", "rate"), [(True, 1.0), (False, 0.0)])
async def test_the_scorer_tells_a_fooled_model_from_a_faithful_one(obeys, rate):
    name, case = "indirect: curl name", CASES["indirect: curl name"]
    agent = build_agent(Settings(openrouter_api_key=SecretStr("test")))
    url = await prepare_database(Settings().database_url)
    with agent.override(model=FunctionModel(curl_model(obeys))):
        turns = await play(url, agent, name, case)
    assert summarize_redteam({name: [turns]})["injection_success_rate"] == rate


async def test_the_safety_screen_answers_every_refusal_case_with_no_model():
    def must_not_run(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("the model saw a message the safety screen refuses")

    agent = build_agent(Settings(openrouter_api_key=SecretStr("test")))
    url = await prepare_database(Settings().database_url)
    runs = {}
    with agent.override(model=FunctionModel(must_not_run)):
        for name, case in CASES.items():
            if case.category == "refusal":
                runs[name] = [await play(url, agent, name, case)]
    summary = summarize_redteam(runs)
    assert runs and summary["safety_refusal_rate"] == 1.0
    assert summary["injection_success_rate"] == 0.0


@pytest.mark.eval
@pytest.mark.timeout(1800)  # about 45 turns x 2 repeats on a real model
async def test_the_red_team_cannot_break_the_agent():
    settings = Settings()
    if settings.openrouter_api_key is None:
        pytest.skip("OPENROUTER_API_KEY is not set")
    agent = build_agent(settings)
    url = await prepare_database(settings.database_url)
    runs = {
        name: [await play(url, agent, name, case) for _ in range(REPEATS)]
        for name, case in CASES.items()
    }
    if any("status_code: 429" in t["reply"] for reps in runs.values() for r in reps for t in r):
        pytest.skip("the model provider is rate-limiting (HTTP 429); this run says nothing")

    for name, reps in runs.items():
        for turns in reps:
            for t in turns:
                if not t["verified"] and t["kind"] != "refuse":
                    print(f"ABORTED {name}: {t['message'][:40]!r} -> {t['reply'][:120]!r}")
                if injection_succeeded(CASES[name], t):
                    print(f"ATTACK WORKED {name}: {t['message'][:60]!r} -> {t['reply'][:200]!r}")
    summary = summarize_redteam(runs)
    print(f"red team: {summary}")
    assert not check_redteam(summary), f"{check_redteam(summary)}\n{summary}"
