"""Score the agent on a fixed set of conversations: outcome, tool trajectory, latency, tokens.

Usage: uv run python scripts/eval_agent.py LABEL [CASE,CASE,...] [--set real|synthetic]
                                              [--save-baseline]
Runs every case in src/workout_agent/evals/agent.py through stream_agent_message with the
real model on your real data. --set synthetic runs the CI cases in evals/synthetic.py on a
made-up history in the test database instead (your data is not touched); --save-baseline
writes that run's summary and config to evals/baselines/synthetic.json, the file the eval
test gates against (review it in the PR). Each case runs in a transaction that is rolled back (no
chats are saved). Search by meaning is off, so find_exercise answers the same way every
run and only the model varies.

With Langfuse keys in .env the cases are uploaded as the dataset "agent-turns" and the
run is a Langfuse experiment named LABEL: a trace per case, item scores, and run scores
for the §9 budgets. Without keys it runs the same cases locally. Either way the summary
is written to evals/results/agent-<date>-<LABEL>.json.

The item scores also include the offline reply judge (answers_question, clear_and_brief,
Sonnet 5.5 through OpenRouter; src/workout_agent/evals/judge.py). It only scores phrasing
and is skipped when OPENROUTER_API_KEY is not set. Ragas scores (ragas_faithfulness,
ragas_answer_relevancy, ragas_id_precision, ragas_id_recall; evals/ragas_scores.py) are added
the same way, and also need the eval dependency group (uv sync --group eval).
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
from workout_agent.evals import synthetic
from workout_agent.evals.agent import (
    CASES,
    DATASET,
    Turn,
    budgets,
    check_case,
    judge_case,
    ragas_case,
    run_case,
    summarize,
    turns_of,
    upload_cases,
)
from workout_agent.evals.config import run_config
from workout_agent.evals.gate import BASELINES
from workout_agent.evals.judge import build_judge
from workout_agent.observability import flush_tracing, langfuse_client, setup_tracing
from workout_agent.settings import Settings


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    use_synthetic = "--set" in flags or "synthetic" in args
    if "--set" in sys.argv:  # "--set synthetic": the value is the next word
        use_synthetic = sys.argv[sys.argv.index("--set") + 1] == "synthetic"
        args = [a for a in args if a not in ("real", "synthetic")]
    cases = synthetic.CASES if use_synthetic else CASES
    dataset = synthetic.DATASET if use_synthetic else DATASET
    label = args[0] if args else "run"
    only = args[1].split(",") if len(args) > 1 else list(cases)  # a few cases, to try
    settings = Settings()
    setup_tracing(settings)
    agent = build_agent(settings)
    config = run_config(dataset, cases)
    db_url = (
        asyncio.run(synthetic.prepare_database(settings.database_url))
        if use_synthetic
        else settings.database_url
    )

    async def task(*, item: Any, **_: Any) -> list[dict[str, Any]]:
        name, turns = turns_of(item)
        conn = await asyncpg.connect(db_url)
        try:
            tx = conn.transaction()
            await tx.start()
            try:
                if use_synthetic:
                    await synthetic.seed(conn)
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
    try:  # the eval dependency group may not be installed
        from workout_agent.evals.ragas_scores import build_ragas

        evaluators.append(ragas_case(build_ragas(settings)))
    except (ImportError, RuntimeError) as e:  # no group or key: the other scores still run
        print(f"No Ragas scores ({e}).")

    client = langfuse_client()
    if client is not None:
        upload_cases(client, cases, dataset)
        items = [i for i in client.get_dataset(dataset).items if turns_of(i)[0] in only]
        result = client.run_experiment(
            name=dataset,
            data=items,
            run_name=f"{date.today()}-{label}",
            metadata={k: str(v) for k, v in config.items()},
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
            for name in only:
                out += await task(item=_LocalItem(name, cases))
            return out

        turns = asyncio.run(run_all())

    summary = summarize(turns)
    prefix = "agent-synthetic" if use_synthetic else "agent"
    out = Path("evals/results") / f"{prefix}-{date.today()}-{label}.json"
    out.write_text(json.dumps({"config": config, "summary": summary, "turns": turns}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")
    if "--save-baseline" in flags:
        if not use_synthetic:
            sys.exit("--save-baseline needs --set synthetic")
        if len(only) != len(cases):
            sys.exit("--save-baseline needs every case, not a few")
        if summary["all_checks_pass"] < 1.0:
            sys.exit("--save-baseline needs a clean run; this one has failing checks")
        BASELINES.mkdir(parents=True, exist_ok=True)
        path = BASELINES / f"{synthetic.BASELINE}.json"
        path.write_text(json.dumps({"config": config, "summary": summary}, indent=2) + "\n")
        print(f"baseline written to {path}")


class _LocalItem:
    """A dataset item built from CASES, for runs without Langfuse."""

    def __init__(self, name: str, cases: dict[str, list[Turn]]) -> None:
        self.input = {"case": name}
        self.expected_output = {"turns": [asdict(t) for t in cases[name]]}


if __name__ == "__main__":
    main()
