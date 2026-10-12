"""Integration test: run_cases drives the real find_exercise over the test database."""

from workout_agent.embeddings.search import ScoredExercise
from workout_agent.evals.search import Case, run_cases, summarize

CASES = [
    Case(phrase="bench press", kind="exact_words", expected=(2,)),
    Case(phrase="chest press", kind="slang_synonym", expected=(2,)),
    Case(phrase="zzz", kind="noise", expected=()),
]

HITS = {
    "bench press": [ScoredExercise(2, "Barbell Bench Press", 0.9)],
    "chest press": [
        ScoredExercise(8, "Barbell Incline Bench Press", 0.56),
        ScoredExercise(2, "Barbell Bench Press", 0.55),
    ],
    "zzz": [ScoredExercise(8, "Barbell Incline Bench Press", 0.2)],
}


async def fake_search(query: str) -> list[ScoredExercise]:
    return HITS[query]


async def test_run_cases_scores_each_step(conn, seeded):
    results = await run_cases(conn, CASES, fake_search)
    by_phrase = {r.phrase: r for r in results}

    # Text match: both bench presses contain the words; the meaning search is not used.
    assert by_phrase["bench press"].step == "text"
    assert by_phrase["bench press"].hit_at_3
    # Raw hits are kept even when the text match answered.
    assert by_phrase["bench press"].best_right_score == 0.9

    # No name contains "chest": falls back to meaning, right answer second.
    assert by_phrase["chest press"].step == "meaning"
    assert not by_phrase["chest press"].hit_at_1
    assert by_phrase["chest press"].hit_at_3

    # Below MIN_SCORE: nothing offered, so the noise case passes.
    assert by_phrase["zzz"].step == "none"
    assert by_phrase["zzz"].passed

    assert summarize(results)["all"].passed == 3


async def test_run_cases_without_search_uses_text_only(conn, seeded):
    results = await run_cases(conn, CASES, None)
    assert [r.step for r in results] == ["text", "none", "none"]
    assert all(r.raw == [] for r in results)
