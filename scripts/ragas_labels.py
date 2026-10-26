"""Score the hand-labelled replies with Ragas answer relevancy and compare with the labels.

Usage: uv run python scripts/ragas_labels.py [--judged evals/results/judge-<...>.json]

Loads the 53 replies in evals/judge_labels.toml (scripts/judge_results.py), scores each
with ragas_answer_relevancy, and prints how well that score separates the hand labels
for answers_question: mean score for yes and no replies, the share of yes/no pairs ranked
the right way (AUC), and agreement with the labels and with the Sonnet judge at a few
cut-offs, next to an always-yes baseline. These replies were saved as text without tool
results, so faithfulness cannot be scored here (see the agent run for that), and Ragas has
no counterpart to clear_and_brief. Offline only, like the judge.
The run is written to evals/results/ragas-<date>-labels.json.
"""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

from judge_results import RESULTS, load_labels
from ragas.metrics.collections import AnswerRelevancy

from workout_agent.evals.ragas_scores import build_ragas
from workout_agent.settings import Settings

CUTOFFS = (0.3, 0.5, 0.7)
SONNET_ON_LABELS = RESULTS / "judge-2026-10-22-real-bad-owner.json"  # Sonnet 5.5, all 53


async def score_all(rows: list[dict[str, Any]]) -> None:
    scorer = build_ragas(Settings())
    metric = AnswerRelevancy(llm=scorer.llm, embeddings=scorer.embeddings)
    gate = asyncio.Semaphore(4)

    async def one(row: dict[str, Any]) -> None:
        async with gate:
            try:
                row["ragas_answer_relevancy"] = (
                    await metric.ascore(user_input=row["message"], response=row["reply"])
                ).value
            except Exception as e:  # recorded, left out of the numbers
                row["ragas_error"] = repr(e)

    await asyncio.gather(*(one(r) for r in rows))


def auc(yes: list[float], no: list[float]) -> float:
    """The share of (yes, no) pairs where the yes reply scores higher; ties count half."""
    pairs = [(y > n) + 0.5 * (y == n) for y in yes for n in no]
    return sum(pairs) / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", type=Path, default=SONNET_ON_LABELS)
    args = parser.parse_args()

    rows = load_labels()
    judged = {
        (r["case"], r["message"], r["reply"]): r["judge"]
        for r in json.loads(args.judged.read_text())["rows"]
    }
    asyncio.run(score_all(rows))
    ok = [r for r in rows if "ragas_answer_relevancy" in r]
    yes = [r["ragas_answer_relevancy"] for r in ok if r["answers_question"]]
    no = [r["ragas_answer_relevancy"] for r in ok if not r["answers_question"]]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"scored {len(ok)} of {len(rows)} ({len(rows) - len(ok)} errors)")
    print(
        f"label yes: n={len(yes)} mean {mean(yes):.2f}; label no: n={len(no)} mean {mean(no):.2f}"
    )
    print(f"AUC (yes ranked above no): {auc(yes, no):.2f}")
    summary: dict[str, Any] = {"n": len(ok), "yes": len(yes), "no": len(no), "auc": auc(yes, no)}
    for cut in CUTOFFS:
        said = {id(r): r["ragas_answer_relevancy"] >= cut for r in ok}
        label = sum(said[id(r)] == r["answers_question"] for r in ok)
        sonnet = [
            said[id(r)] == judged[(r["case"], r["message"], r["reply"])]["answers_question"]
            for r in ok
            if (r["case"], r["message"], r["reply"]) in judged
        ]
        summary[f">={cut}"] = {"label": label, "sonnet": sum(sonnet), "sonnet_of": len(sonnet)}
        print(
            f">={cut}: labels {label}/{len(ok)}, Sonnet {sum(sonnet)}/{len(sonnet)} agree"
        )
    print(f"always-yes agrees with labels {len(yes)}/{len(ok)}")
    print(f"\n{'case':<24}{'message':<34}{'label':<6}{'relevancy':<10}")
    for r in sorted(ok, key=lambda r: r["ragas_answer_relevancy"]):
        print(
            f"{r['case'][:23]:<24}{r['message'][:33]:<34}"
            f"{'yes' if r['answers_question'] else 'no':<6}{r['ragas_answer_relevancy']:.2f}"
        )
    out = RESULTS / f"ragas-{date.today()}-labels.json"
    keep = ("case", "message", "reply", "answers_question", "ragas_answer_relevancy", "ragas_error")
    out.write_text(
        json.dumps(
            {"summary": summary, "rows": [{k: r[k] for k in keep if k in r} for r in rows]},
            indent=2,
        )
    )
    print(f"written to {out}")


if __name__ == "__main__":
    main()
