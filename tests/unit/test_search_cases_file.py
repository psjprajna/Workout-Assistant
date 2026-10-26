"""The checked-in evals/search_cases.toml loads and covers every kind."""

from pathlib import Path

from workout_agent.evals.search import KINDS, load_cases

CASES = Path(__file__).resolve().parents[2] / "evals" / "search_cases.toml"


def test_cases_file_loads_and_covers_every_kind():
    cases = load_cases(CASES)
    assert {c.kind for c in cases} == set(KINDS)
    assert len(cases) >= 60


def test_cases_file_has_a_holdout_split_covering_every_kind():
    holdout = [c for c in load_cases(CASES) if c.split == "holdout"]
    assert {c.kind for c in holdout} == set(KINDS)
