"""Judge the agent's replies for phrasing and compare the judge with the hand labels.

Usage:
  uv run python scripts/judge_results.py --judge typesafe:jev-latest
  uv run python scripts/judge_results.py --judge openrouter            # default model below
  uv run python scripts/judge_results.py --judge openrouter:<slug> --label my-run
  uv run python scripts/judge_results.py --judge openrouter --results evals/results/<run>.json

Default mode judges the 24 replies in evals/judge_labels.toml and prints how often the
judge agrees with the hand labels, next to a judge that always says yes. With --results it
judges every non-refusal turn of an agent run and prints the verdicts only. Offline: the
judge never runs in the request path and nothing here touches the agent or the database.
The run is written to evals/results/judge-<date>-<label>.json.
"""

import argparse
import asyncio
import json
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from workout_agent.evals.judge import QUESTIONS, Verdict, build_judge, judge_one
from workout_agent.settings import Settings

RESULTS = Path("evals/results")
LABELS = Path("evals/judge_labels.toml")


def load_labels() -> list[dict[str, Any]]:
    """The hand labels, each with its reply: inline for made-up, else from the results file."""
    with LABELS.open("rb") as f:
        entries = tomllib.load(f)["case"]
    runs: dict[str, list[dict[str, Any]]] = {}
    out = []
    for e in entries:
        if e["source"] == "made-up":
            reply = e["reply"]
        else:
            turns = runs.setdefault(
                e["source"], json.loads((RESULTS / e["source"]).read_text())["turns"]
            )
            found = [t for t in turns if t["case"] == e["case"] and t["message"] == e["message"]]
            if len(found) != 1:
                raise SystemExit(f"{len(found)} replies for {e['case']!r} / {e['message']!r}")
            reply = found[0]["reply"]
        out.append({**e, "reply": reply})
    return out


def load_turns(path: Path) -> list[dict[str, Any]]:
    """Every turn of an agent run that is not a refusal."""
    return [t for t in json.loads(path.read_text())["turns"] if t["kind"] != "refuse"]


def yn(value: bool | None) -> str:
    return "-" if value is None else "yes" if value else "no"


def report_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"{'case':<24}{'message':<34}{'judge aq/cb':<13}{'label aq/cb':<13}")
    for r in rows:
        j = r["judge"]
        judged = "error" not in j
        jv = f"{yn(j['answers_question'])}/{yn(j['clear_and_brief'])}" if judged else "ERR"
        lv = f"{yn(r['answers_question'])}/{yn(r['clear_and_brief'])}"
        match = "ERR" if not judged else "MATCH" if all(j[q] == r[q] for q in QUESTIONS) else "DIFF"
        print(f"{r['case'][:23]:<24}{r['message'][:33]:<34}{jv:<13}{lv:<13}{match}")
    ok = [r for r in rows if "error" not in r["judge"]]
    summary: dict[str, Any] = {"judged": len(ok), "errors": len(rows) - len(ok)}
    total = agree_total = yes_total = 0
    for q in QUESTIONS:
        agree = sum(r["judge"][q] == r[q] for r in ok)
        yes = sum(r[q] for r in ok)  # an always-yes judge is right wherever the label is yes
        summary[q] = {"agree": agree, "always_yes": yes, "of": len(ok)}
        total += len(ok)
        agree_total += agree
        yes_total += yes
        print(f"{q}: judge {agree}/{len(ok)}, always-yes {yes}/{len(ok)}")
    summary["overall"] = {"agree": agree_total, "always_yes": yes_total, "of": total}
    print(f"overall: judge {agree_total}/{total}, always-yes {yes_total}/{total}")
    if summary["errors"]:
        print(f"{summary['errors']} errored rows are left out of the counts above")
    return summary


def report_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"{'case':<24}{'message':<34}{'aq':<5}{'cb':<5}")
    for r in rows:
        j = r["judge"]
        aq, cb = (yn(j.get(q)) if "error" not in j else "ERR" for q in QUESTIONS)
        print(f"{r['case'][:23]:<24}{r['message'][:33]:<34}{aq:<5}{cb:<5}")
    return {"turns": len(rows), "errors": sum("error" in r["judge"] for r in rows)}


async def run(agent: Agent[None, Verdict], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**i, "judge": await judge_one(agent, i["message"], i["reply"])} for i in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge agent replies for phrasing.")
    parser.add_argument("--judge", required=True, help="typesafe:jev-latest or openrouter[:slug]")
    parser.add_argument("--results", type=Path, help="judge an agent run instead of the labels")
    parser.add_argument("--label", help="name for the output file (default: the judge)")
    args = parser.parse_args()

    try:
        agent = build_judge(args.judge, Settings())
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    items = load_turns(args.results) if args.results else load_labels()
    rows = asyncio.run(run(agent, items))
    summary = report_results(rows) if args.results else report_labels(rows)

    label = args.label or re.sub(r"[^a-z0-9]+", "-", args.judge.lower()).strip("-")
    out = RESULTS / f"judge-{date.today()}-{label}.json"
    keep = ("case", "message", "reply", "judge", *QUESTIONS)
    out.write_text(
        json.dumps(
            {
                "judge": args.judge,
                "mode": "results" if args.results else "labels",
                "summary": summary,
                "rows": [{k: r[k] for k in keep if k in r} for r in rows],
            },
            indent=2,
        )
    )
    print(f"written to {out}")


if __name__ == "__main__":
    main()
