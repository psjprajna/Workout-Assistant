"""Replay find_exercise's decide() over a saved eval run for a range of cutoffs.

Usage: uv run python scripts/sweep_search.py evals/results/search-<date>-<label>.json
No database, no API: every stage's candidates are saved in the results file, so this
shows what each MIN_SCORE would have done. Pick the cutoff from the dev rows; the
holdout rows show whether it carries over.
"""

import argparse
import json
import sys
from pathlib import Path

from workout_agent.evals.search import (
    candidates_from_dict,
    case_from_dict,
    score_candidates,
    summarize,
)


def main(path: Path, low: float, high: float, step: float) -> int:
    record = json.loads(path.read_text())
    cases = [
        (case_from_dict(c), candidates_from_dict(c["candidates"]))
        for c in record["cases"]
        if c.get("candidates")
    ]
    if not cases:
        print("no saved candidates in this file: rerun scripts/eval_search.py")
        return 1

    # Lowest score a right answer reached and highest a noise phrase reached, by meaning.
    for split in ("dev", "holdout"):
        right = [
            max((h.score for h in cand.meaning if h.exercise_id in case.expected), default=0)
            for case, cand in cases
            if case.split == split and case.expected
        ]
        noise = [
            max((h.score for h in cand.meaning), default=0)
            for case, cand in cases
            if case.split == split and not case.expected and not cand.query.not_an_exercise
        ]
        print(f"{split}: noise tops out at {max(noise, default=0):.3f}; "
              f"right answers by meaning: {sorted(round(r, 3) for r in right)[:8]} …")

    print(f"\n{'min_score':>9}  {'dev passed':>10} {'hit@1':>5} {'wrong_ok':>8} {'false':>5}"
          f"   {'holdout passed':>14} {'hit@1':>5} {'wrong_ok':>8} {'false':>5}")
    steps = round((high - low) / step) + 1
    for i in range(steps):
        cutoff = round(low + i * step, 3)
        results = [score_candidates(case, cand, min_score=cutoff) for case, cand in cases]
        s = summarize(results)
        dev, hold = s.get("dev"), s.get("holdout")
        row = f"{cutoff:>9.2f}"
        for t in (dev, hold):
            if t is None:
                row += f"  {'-':>10} {'-':>5} {'-':>8} {'-':>5} "
            else:
                row += (f"  {t.passed:>4}/{t.cases:<5} {t.hit_at_1:>5} {t.wrong_ok:>8}"
                        f" {t.false_offers:>5} ")
        print(row)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay decide() for a range of cutoffs.")
    parser.add_argument("results", type=Path, help="a results file from eval_search.py")
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    sys.exit(main(args.results, args.low, args.high, args.step))
