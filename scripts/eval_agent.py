"""Score the agent on a fixed set of conversations: outcome, tool trajectory, latency, tokens.

Usage: uv run python scripts/eval_agent.py LABEL
Runs every case in src/workout_agent/evals/agent.py through stream_agent_message with the
real model on your real data. Each case runs in a transaction that is rolled back (no
chats are saved). Search by meaning is off, so find_exercise answers the same way every
run and only the model varies.

With Langfuse keys in .env the cases are uploaded as the dataset "agent-turns" and the
run is a Langfuse experiment named LABEL: a trace per case, item scores, and run scores
for the §9 budgets. Without keys it runs the same cases locally. Either way the summary
is written to evals/results/agent-<date>-<LABEL>.json.

The item scores also include the offline reply judge (answers_question, clear_and_brief,
Sonnet 5.5 through OpenRouter; src/workout_agent/evals/judge.py). It only scores phrasing
and is skipped when OPENROUTER_API_KEY is not set.
"""

import asyncio
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import asyncpg
from langfuse.experiment import EvaluatorFunction

from workout_agent.agents.workout_agent import build_agent
from workout_agent.evals.agent import (
    CASES,
    DATASET,
    budgets,
    check_case,
    judge_case,
    run_case,
    summarize,
    turns_of,
    upload_cases,
)
from workout_agent.evals.judge import build_judge
from workout_agent.observability import flush_tracing, langfuse_client, setup_tracing
from workout_agent.settings import Settings


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    settings = Settings()
    setup_tracing(settings)
    agent = build_agent(settings)

    async def task(*, item: Any, **_: Any) -> list[dict[str, Any]]:
        name, turns = turns_of(item)
        conn = await asyncpg.connect(settings.database_url)
        try:
            tx = conn.transaction()
            await tx.start()
            try:
                return [asdict(r) for r in await run_case(conn, agent, name, turns)]
            finally:
                await tx.rollback()  # leave no eval chats behind
        finally:
            await conn.close()

    evaluators: list[EvaluatorFunction] = [check_case]
    try:
        evaluators.append(judge_case(build_judge("openrouter", settings)))
    except RuntimeError as e:  # no key: the code checks still run
        print(f"No judge scores ({e}).")

    client = langfuse_client()
    if client is not None:
        upload_cases(client)
        result = client.get_dataset(DATASET).run_experiment(
            name=DATASET,
            run_name=f"{date.today()}-{label}",
            task=task,
            evaluators=evaluators,
            run_evaluators=[budgets],
            max_concurrency=1,  # one case at a time, so latencies are not skewed
        )
        turns = [t for r in result.item_results for t in (r.output or [])]
        flush_tracing()
        print(result.experiment_url or "")
    else:
        print("Langfuse keys not set: running locally, no traces.")

        async def run_all() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for name in CASES:
                out += await task(item=_LocalItem(name))
            return out

        turns = asyncio.run(run_all())

    summary = summarize(turns)
    out = Path("evals/results") / f"agent-{date.today()}-{label}.json"
    out.write_text(json.dumps({"summary": summary, "turns": turns}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")


class _LocalItem:
    """A dataset item built from CASES, for runs without Langfuse."""

    def __init__(self, name: str) -> None:
        self.input = {"case": name}
        self.expected_output = {"turns": [asdict(t) for t in CASES[name]]}


if __name__ == "__main__":
    main()
