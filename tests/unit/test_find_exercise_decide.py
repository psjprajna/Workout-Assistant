"""Unit tests: parse_query() and decide(), the pure parts of find_exercise (no database)."""

from workout_agent.embeddings.search import ScoredExercise
from workout_agent.tools.find_exercise import (
    Candidates,
    TextHit,
    Thresholds,
    decide,
    parse_query,
)


def _text(*ids, exact=False):
    return [TextHit(i, f"Exercise {i}", exact, usage_count=0) for i in ids]


def _scored(*pairs):
    return [ScoredExercise(i, f"Exercise {i}", score) for i, score in pairs]


def _ids(decision):
    return [m.exercise_id for m in decision.result.matches]


def test_parse_query_tidies_and_expands_shorthand():
    q = parse_query("  DB Bench-Press ")
    assert q.phrase == "dumbbell bench press"
    assert q.words == ["dumbbell", "bench", "press"]


def test_parse_query_drops_filler_and_spots_muscle_phrases():
    q = parse_query("something for my lower back")
    assert q.words == ["lower", "back"]
    assert q.muscle_only
    assert not parse_query("back extension").muscle_only


def test_parse_query_spots_days_and_workouts():
    assert parse_query("leg day").not_an_exercise
    assert parse_query("my workout").not_an_exercise
    assert not parse_query("leg press").not_an_exercise


def test_one_exact_name_is_ok_even_with_partial_matches():
    c = Candidates(parse_query("leg press"), text=_text(132, exact=True) + _text(1102))
    d = decide(c)
    assert (d.result.status, d.step, _ids(d)) == ("ok", "exact", [132])


def test_one_partial_match_is_ok():
    d = decide(Candidates(parse_query("hip thrust"), text=_text(386)))
    assert (d.result.status, _ids(d)) == ("ok", [386])


def test_muscle_phrase_with_one_partial_match_is_ambiguous_with_meanings():
    c = Candidates(
        parse_query("forearms"), text=_text(627), meaning=_scored((119, 0.5), (627, 0.49))
    )
    d = decide(c)
    assert (d.result.status, _ids(d)) == ("ambiguous", [627, 119])


def test_spelling_fixes_come_before_meanings_and_are_capped_at_three():
    c = Candidates(
        parse_query("dumbell rows"),
        text=[],
        fuzzy=_text(88, 38),
        meaning=_scored((88, 0.6), (3, 0.5), (52, 0.45)),
    )
    d = decide(c)
    assert (d.result.status, d.step, _ids(d)) == ("ambiguous", "fuzzy", [88, 38, 3])


def test_meanings_below_the_cutoff_are_not_offered():
    c = Candidates(parse_query("zzz"), text=[], meaning=_scored((1, 0.3)))
    assert decide(c, Thresholds(min_score=0.4)).result.status == "no_rows"
    assert decide(c, Thresholds(min_score=0.25)).result.status == "ambiguous"


def test_not_an_exercise_wins_over_everything():
    c = Candidates(parse_query("push day"), text=_text(47), meaning=_scored((47, 0.9)))
    d = decide(c)
    assert (d.result.status, d.step) == ("no_rows", "guard")


def test_one_exact_alias_is_ok_only_when_nothing_else_matches():
    alias = [TextHit(199, "Barbell Romanian Deadlift", False, 0, exact_alias=True)]
    alone = decide(Candidates(parse_query("rdl"), text=alias))
    assert (alone.result.status, _ids(alone)) == ("ok", [199])

    crowded = decide(Candidates(parse_query("squat"), text=alias + _text(135)))
    assert (crowded.result.status, _ids(crowded)) == ("ambiguous", [199, 135])
