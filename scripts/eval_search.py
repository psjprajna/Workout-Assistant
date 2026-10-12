"""Score find_exercise on evals/search_cases.toml and save the results.

Usage: uv run python scripts/eval_search.py --label NAME [--compare FILE] [--model MODEL]
Needs OPENROUTER_API_KEY in .env, Postgres and Qdrant running, the export imported and
scripts/index_exercises.py run. Writes evals/results/search-<date>-<label>.json; commit it
so later changes can be compared with --compare.
"""

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from functools import partial
from pathlib import Path

import asyncpg
from qdrant_client import AsyncQdrantClient

from workout_agent.db.migrate import apply_migrations
from workout_agent.embeddings.cache import cached
from workout_agent.embeddings.index import COLLECTION
from workout_agent.embeddings.openrouter import MODEL_NAME, collection_for, embed
from workout_agent.embeddings.search import MIN_SCORE, search_exercises
from workout_agent.evals.search import (
    CaseResult,
    Tally,
    compare,
    load_cases,
    meaning_only,
    run_cases,
    summarize,
)
from workout_agent.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "search_cases.toml"
RESULTS = ROOT / "evals" / "results"

# Raw Qdrant hits kept per phrase; find_exercise still offers at most 3.
RAW_LIMIT = 5


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def _pct(part: int, whole: int) -> str:
    return f"{part:>3}/{whole:<3} {100 * part / whole:5.1f}%" if whole else "-"


def _print_summary(summary: dict[str, Tally]) -> None:
    print(f"\n{'kind':<14} {'passed':>14} {'hit@1':>14} {'hit@3':>14} wrong_ok false_offer  steps")
    for kind, t in summary.items():
        steps = ", ".join(f"{k} {v}" for k, v in sorted(t.steps.items()))
        print(
            f"{kind:<14} {_pct(t.passed, t.cases):>14} {_pct(t.hit_at_1, t.cases):>14} "
            f"{_pct(t.hit_at_3, t.cases):>14} {t.wrong_ok:>8} {t.false_offers:>11}  {steps}"
        )


def _print_failures(results: list[CaseResult], names: dict[int, str]) -> None:
    failed = [r for r in results if not r.passed]
    print(f"\nFailed cases ({len(failed)}):")
    for r in failed:
        shown = ", ".join(o.name for o in r.options[:3]) or "nothing"
        wanted = ", ".join(names[i] for i in r.expected[:3]) or "nothing"
        top = r.raw[0] if r.raw else None
        top_text = f"{top.score:.3f} {top.name}" if top else "-"
        tag = r.kind if r.split == "dev" else f"{r.kind}, holdout"
        print(f"  [{tag}] {r.phrase!r}: {r.step}/{r.status} -> {shown}")
        print(f"      wanted {wanted}; top raw {top_text}; best right {r.best_right_score}")


async def main(label: str, compare_with: Path | None, model: str) -> int:
    settings = Settings()
    collection = collection_for(model, COLLECTION)
    cases = load_cases(CASES)

    async def fetch(texts: list[str]) -> list[list[float]]:
        return await embed(settings, texts, model)

    conn = await asyncpg.connect(settings.database_url)
    await apply_migrations(conn)
    embedder = cached(conn, model, fetch)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        names = {
            r["exercise_id"]: r["name"]
            for r in await conn.fetch("select exercise_id, name from exercises")
        }
        unknown = sorted({i for c in cases for i in c.expected} - names.keys())
        if unknown:
            print(f"expected ids not in the exercises table: {unknown}")
            return 1
        if not await qdrant.collection_exists(collection):
            print(f"Qdrant has no {collection} collection: run scripts/index_exercises.py")
            return 1
        points = (await qdrant.count(collection, exact=True)).count

        search = partial(
            search_exercises, qdrant, embedder, limit=RAW_LIMIT, collection=collection
        )
        results = await run_cases(conn, cases, search)
    finally:
        await conn.close()
        await qdrant.close()

    summary = summarize(results)
    _print_summary(summary)
    meaning = {split: meaning_only(results, split) for split in ("dev", "holdout")}
    print("\nQdrant alone (raw top hits, text stages ignored):")
    for split, m in meaning.items():
        print(
            f"  {split:<8} hit@1 {_pct(m.hit_at_1, m.cases)}  hit@3 {_pct(m.hit_at_3, m.cases)}"
            f"  noise max {m.noise_max:.3f}  lowest right {m.right_min}"
        )
    _print_failures(results, names)

    RESULTS.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9-]+", "-", label.lower()).strip("-")
    out = RESULTS / f"search-{date.today().isoformat()}-{slug}.json"
    record = {
        "label": label,
        "date": date.today().isoformat(),
        "settings": {
            "embedding_model": model,
            "min_score": MIN_SCORE,
            "qdrant_points": points,
            "db_exercises": len(names),
            "git_commit": _git_commit(),
            "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        },
        "summary": {kind: asdict(t) for kind, t in summary.items()},
        "meaning_only": {split: asdict(m) for split, m in meaning.items()},
        "cases": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")

    if compare_with is not None:
        old = json.loads(compare_with.read_text())
        if old["settings"]["cases_sha256"] != record["settings"]["cases_sha256"]:
            print("note: the cases file changed since that run; only shared phrases compared")
        fixed, broken = compare(
            {c["phrase"]: c["passed"] for c in old["cases"]},
            {r.phrase: r.passed for r in results},
        )
        old_all, new_all = old["summary"]["all"], record["summary"]["all"]
        print(f"\nvs {compare_with.name}:")
        for key in ("passed", "hit_at_1", "hit_at_3", "wrong_ok", "false_offers"):
            print(f"  {key:<13} {old_all[key]:>3} -> {new_all[key]:>3}")
        print(f"  fixed:  {fixed or '-'}")
        print(f"  broken: {broken or '-'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score find_exercise on the labelled phrases.")
    parser.add_argument("--label", required=True, help="short name for this run")
    parser.add_argument("--compare", type=Path, help="an earlier results file")
    parser.add_argument("--model", default=MODEL_NAME, help="embedding model to evaluate")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.label, args.compare, args.model)))
