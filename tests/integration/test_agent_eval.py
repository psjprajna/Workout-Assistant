"""Agent evals as tests (docs/amendments.md §25).

The synthetic cases (workout_agent.evals.synthetic) run through the agent and the run is held
to the §9 budgets and to the committed baseline (evals/baselines/synthetic.json).

- test_a_scripted_run_...: a FunctionModel stands in for the model, so it runs everywhere and
  proves the cases, the seed, run_case and the gate fit together.
- test_the_synthetic_cases_pass_the_gate (marker `eval`): the real model. Each case runs
  three times and passes on a majority; the whole run is then checked by check_gate.
  Set EVAL_TIMING=1 to also gate latency (the scheduled run does; PR runs do not).
"""

import os
from collections import defaultdict
from dataclasses import asdict
from typing import Any

import asyncpg
import pytest
from pydantic import SecretStr
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.agents.workout_agent import build_agent
from workout_agent.evals.agent import FIND, LAST, Turn, run_case, summarize
from workout_agent.evals.gate import check_gate, load_baseline
from workout_agent.evals.synthetic import BASELINE, CASES, prepare_database, seed
from workout_agent.settings import Settings

REPEATS = 3


async def play(url: str, agent: Any, name: str, turns: list[Turn]) -> list[dict[str, Any]]:
    """One case on the seeded history, in a transaction that is rolled back."""
    conn = await asyncpg.connect(url)
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await seed(conn)
            return [asdict(r) for r in await run_case(conn, agent, name, turns)]
        finally:
            await tx.rollback()
    finally:
        await conn.close()


async def test_a_scripted_run_over_the_synthetic_seed_fetches_and_summarizes():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart(FIND, {"query": "barbell curl"})])
        if returns[0].tool_name == FIND:
            return ModelResponse(parts=[ToolCallPart(LAST, {"exercise_id": 20})])
        return ModelResponse(
            parts=[ToolCallPart("final_result_ShowLastSession", {"exercise_id": 20})]
        )

    agent = build_agent(Settings(openrouter_api_key=SecretStr("test")))
    url = await prepare_database(Settings().database_url)
    turns = CASES["barbell curl"]
    with agent.override(model=FunctionModel(model)):
        out = await play(url, agent, "barbell curl", turns)

    assert out[0]["errors"] == []
    summary = summarize(out)
    assert summary["kind_accuracy"] == 1.0
    assert check_gate(summary, summary) == []  # a run is never a regression from itself


@pytest.mark.eval
@pytest.mark.timeout(1200)  # 17 turns x 3 repeats on a real model
async def test_the_synthetic_cases_pass_the_gate():
    settings = Settings()
    if settings.openrouter_api_key is None:
        pytest.skip("OPENROUTER_API_KEY is not set")
    baseline = load_baseline(BASELINE)["summary"]
    agent = build_agent(settings)
    url = await prepare_database(settings.database_url)

    runs: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for name, turns in CASES.items():
        for _ in range(REPEATS):
            runs[name].append(await play(url, agent, name, turns))

    if any("status_code: 429" in t["reply"] for reps in runs.values() for r in reps for t in r):
        pytest.skip("the model provider is rate-limiting (HTTP 429); this run says nothing")

    failed = {
        name: [e for r in reps for t in r for e in t["errors"]]
        for name, reps in runs.items()
        if sum(all(not t["errors"] for t in r) for r in reps) <= REPEATS // 2  # no majority
    }
    all_turns = [t for reps in runs.values() for r in reps for t in r]
    problems = check_gate(
        summarize(all_turns), baseline, timing=bool(os.environ.get("EVAL_TIMING"))
    )

    assert not failed, f"cases without a majority of clean runs: {failed}"
    assert not problems, "\n".join(problems)
