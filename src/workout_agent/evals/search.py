"""Score find_exercise against a labelled set of phrases (evals/search_cases.toml).

Scores the whole find_exercise (text match, then search by meaning), since that is what
the user sees, and keeps Qdrant's raw top hits per phrase so MIN_SCORE can be checked.
scripts/eval_search.py runs it against the real database and saves the results.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import asyncpg

from workout_agent.embeddings.search import ScoredExercise
from workout_agent.tools.find_exercise import FindResult, Searcher, find_exercise

KINDS = (
    "exact_words",  # control: the text match should already find these
    "plural_typo",  # "bicep curls", "benchpress"
    "abbreviation",  # "rdl", "db curl"
    "slang_synonym",  # "chest press", "pec deck"
    "muscle_goal",  # "rear delts", "something for triceps"
    "noise",  # nothing should be offered
)

Step = Literal["text", "meaning", "none"]


@dataclass(frozen=True)
class Case:
    phrase: str
    kind: str
    # Exercise ids that count as a right answer. Empty: nothing should be offered.
    expected: tuple[int, ...]


@dataclass
class Option:
    exercise_id: int
    name: str


@dataclass
class RawHit:
    exercise_id: int
    name: str
    score: float


@dataclass
class CaseResult:
    phrase: str
    kind: str
    expected: list[int]
    step: Step  # which part of find_exercise produced the options
    status: str
    options: list[Option]
    hit_at_1: bool  # a right answer is the first option
    hit_at_3: bool  # a right answer is among the first three options
    wrong_ok: bool  # status ok, but on the wrong exercise: the worst failure
    false_offer: bool  # nothing should be offered, but something was
    passed: bool  # hit_at_3, or for a noise case, no offer
    raw: list[RawHit]  # Qdrant's top hits before MIN_SCORE, best first
    best_right_score: float | None
    best_wrong_score: float | None


@dataclass
class Tally:
    cases: int = 0
    passed: int = 0
    hit_at_1: int = 0
    hit_at_3: int = 0
    wrong_ok: int = 0
    false_offers: int = 0
    steps: dict[str, int] = field(default_factory=dict)


def parse_cases(data: Mapping[str, Any]) -> list[Case]:
    """Turn the parsed TOML into cases; raise ValueError on anything malformed."""
    unknown_top = set(data) - {"case"}
    if unknown_top:
        raise ValueError(f"unknown top-level keys: {sorted(unknown_top)}")

    cases: list[Case] = []
    seen: set[str] = set()
    for number, raw in enumerate(data.get("case", []), start=1):
        unknown = set(raw) - {"phrase", "kind", "expected"}
        if unknown:
            raise ValueError(f"case {number}: unknown keys {sorted(unknown)}")
        phrase = raw.get("phrase")
        kind = raw.get("kind")
        expected = raw.get("expected")
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError(f"case {number}: phrase must be a non-empty string")
        if kind not in KINDS:
            raise ValueError(f"case {number} ({phrase!r}): unknown kind {kind!r}")
        if not isinstance(expected, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in expected
        ):
            raise ValueError(f"case {number} ({phrase!r}): expected must be a list of ids")
        if (kind == "noise") != (not expected):
            raise ValueError(
                f"case {number} ({phrase!r}): noise cases, and only they, expect nothing"
            )
        key = phrase.strip().lower()
        if key in seen:
            raise ValueError(f"case {number}: duplicate phrase {phrase!r}")
        seen.add(key)
        cases.append(Case(phrase=phrase, kind=kind, expected=tuple(expected)))

    if not cases:
        raise ValueError("no cases")
    return cases


def load_cases(path: Path) -> list[Case]:
    with path.open("rb") as file:
        return parse_cases(tomllib.load(file))


def score_case(
    case: Case, result: FindResult, searched: bool, raw: list[ScoredExercise]
) -> CaseResult:
    """Score one find_exercise result. `searched`: the meaning fallback was called."""
    expected = set(case.expected)
    ids = [m.exercise_id for m in result.matches]

    step: Step
    if not ids:
        step = "none"
    elif searched:
        step = "meaning"
    else:
        step = "text"

    hit_at_1 = any(i in expected for i in ids[:1])
    hit_at_3 = any(i in expected for i in ids[:3])
    false_offer = not expected and bool(ids)
    right = [hit.score for hit in raw if hit.exercise_id in expected]
    wrong = [hit.score for hit in raw if hit.exercise_id not in expected]

    return CaseResult(
        phrase=case.phrase,
        kind=case.kind,
        expected=list(case.expected),
        step=step,
        status=result.status,
        options=[Option(m.exercise_id, m.name) for m in result.matches],
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        wrong_ok=result.status == "ok" and not hit_at_1,
        false_offer=false_offer,
        passed=not false_offer if not expected else hit_at_3,
        raw=[RawHit(h.exercise_id, h.name, round(h.score, 4)) for h in raw],
        best_right_score=round(max(right), 4) if right else None,
        best_wrong_score=round(max(wrong), 4) if wrong else None,
    )


async def run_cases(
    conn: asyncpg.Connection, cases: list[Case], search: Searcher | None
) -> list[CaseResult]:
    """Run find_exercise on every case.

    `search` is called once per phrase up front for the raw scores; find_exercise then
    gets that same answer, so each phrase costs one embedding call.
    """
    results = []
    for case in cases:
        raw = await search(case.phrase) if search is not None else []
        searched = False

        async def recorded(_query: str, raw: list[ScoredExercise] = raw) -> list[ScoredExercise]:
            nonlocal searched
            searched = True
            return raw

        result = await find_exercise(conn, case.phrase, recorded if search else None)
        results.append(score_case(case, result, searched, raw))
    return results


def summarize(results: list[CaseResult]) -> dict[str, Tally]:
    """Counts overall ("all") and per kind, in KINDS order."""
    tallies = {"all": Tally()} | {kind: Tally() for kind in KINDS}
    for r in results:
        for tally in (tallies["all"], tallies[r.kind]):
            tally.cases += 1
            tally.passed += r.passed
            tally.hit_at_1 += r.hit_at_1
            tally.hit_at_3 += r.hit_at_3
            tally.wrong_ok += r.wrong_ok
            tally.false_offers += r.false_offer
            tally.steps[r.step] = tally.steps.get(r.step, 0) + 1
    return {kind: tally for kind, tally in tallies.items() if tally.cases}


def compare(old: Mapping[str, bool], new: Mapping[str, bool]) -> tuple[list[str], list[str]]:
    """Phrases that went from fail to pass (fixed) and pass to fail (broken).

    Both maps are phrase -> passed; phrases in only one run are ignored.
    """
    shared = [phrase for phrase in new if phrase in old]
    fixed = [p for p in shared if new[p] and not old[p]]
    broken = [p for p in shared if old[p] and not new[p]]
    return fixed, broken
