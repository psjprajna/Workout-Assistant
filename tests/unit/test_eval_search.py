"""Unit tests for the search eval scoring (no database, no embeddings)."""

import pytest

from workout_agent.embeddings.search import ScoredExercise
from workout_agent.evals.search import (
    Case,
    CaseResult,
    compare,
    parse_cases,
    score_case,
    summarize,
)
from workout_agent.tools.find_exercise import ExerciseMatch, FindResult


def _found(status, *ids):
    return FindResult(status=status, matches=[ExerciseMatch(i, f"Exercise {i}") for i in ids])


def _raw(*pairs):
    return [ScoredExercise(i, f"Exercise {i}", score) for i, score in pairs]


def _case(expected=(2,), kind="slang_synonym"):
    return Case(phrase="chest press", kind=kind, expected=tuple(expected))


def test_text_match_on_the_right_exercise_passes():
    result = score_case(_case(), _found("ok", 2), searched=False, raw=[])
    assert result.step == "text"
    assert result.hit_at_1 and result.hit_at_3 and result.passed
    assert not result.wrong_ok


def test_meaning_hit_in_second_place_is_hit_at_3_only():
    result = score_case(_case(), _found("ambiguous", 5, 2, 9), searched=True, raw=[])
    assert result.step == "meaning"
    assert not result.hit_at_1
    assert result.hit_at_3 and result.passed


def test_right_answer_in_fourth_place_fails():
    result = score_case(_case(), _found("ambiguous", 5, 6, 7, 2), searched=False, raw=[])
    assert not result.hit_at_3 and not result.passed


def test_ok_on_the_wrong_exercise_is_wrong_ok():
    result = score_case(_case(), _found("ok", 9), searched=False, raw=[])
    assert result.wrong_ok and not result.passed


def test_nothing_found_is_step_none():
    result = score_case(_case(), _found("no_rows"), searched=True, raw=[])
    assert result.step == "none" and not result.passed


def test_noise_with_no_offer_passes():
    case = _case(expected=(), kind="noise")
    result = score_case(case, _found("no_rows"), searched=True, raw=_raw((7, 0.28)))
    assert result.passed and not result.false_offer


def test_noise_with_an_offer_is_a_false_offer():
    case = _case(expected=(), kind="noise")
    result = score_case(case, _found("ambiguous", 7), searched=True, raw=_raw((7, 0.44)))
    assert result.false_offer and not result.passed


def test_best_right_and_wrong_scores_come_from_raw_hits():
    raw = _raw((9, 0.556), (2, 0.555), (4, 0.30))
    result = score_case(_case(expected=(2, 4)), _found("no_rows"), searched=True, raw=raw)
    assert result.best_right_score == 0.555
    assert result.best_wrong_score == 0.556
    assert [h.exercise_id for h in result.raw] == [9, 2, 4]


def test_summarize_counts_overall_and_per_kind():
    results: list[CaseResult] = [
        score_case(_case(), _found("ok", 2), searched=False, raw=[]),
        score_case(_case(), _found("ok", 9), searched=False, raw=[]),
        score_case(_case((), "noise"), _found("ambiguous", 7), searched=True, raw=[]),
    ]
    summary = summarize(results)
    assert list(summary) == ["all", "slang_synonym", "noise"]
    assert summary["all"].cases == 3
    assert summary["all"].passed == 1
    assert summary["all"].wrong_ok == 1
    assert summary["slang_synonym"].hit_at_1 == 1
    assert summary["noise"].false_offers == 1
    assert summary["all"].steps == {"text": 2, "meaning": 1}


def test_compare_finds_fixed_and_broken_phrases():
    old = {"a": False, "b": True, "c": True, "gone": False}
    new = {"a": True, "b": False, "c": True, "added": False}
    assert compare(old, new) == (["a"], ["b"])


def _toml(**case):
    return {"case": [{"phrase": "rdl", "kind": "abbreviation", "expected": [199]} | case]}


def test_parse_cases_reads_a_valid_case():
    [case] = parse_cases(_toml())
    assert case == Case(phrase="rdl", kind="abbreviation", expected=(199,))


@pytest.mark.parametrize(
    "data",
    [
        {"cases": []},
        {"case": []},
        _toml(phrase="  "),
        _toml(kind="typo"),
        _toml(expected=["199"]),
        _toml(expected=[]),
        _toml(kind="noise"),
        _toml(extra=1),
        {"case": _toml()["case"] * 2},
    ],
    ids=[
        "unknown top key",
        "no cases",
        "blank phrase",
        "unknown kind",
        "non-int id",
        "empty expected outside noise",
        "noise with expected ids",
        "unknown case key",
        "duplicate phrase",
    ],
)
def test_parse_cases_rejects_bad_input(data):
    with pytest.raises(ValueError):
        parse_cases(data)
