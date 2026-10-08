"""Unit tests: build_plan (kg conversion, set numbers, name cleaning, timed exercises)."""

from workout_agent.importing.loader import build_plan
from workout_agent.importing.parsers import ExerciseLog


def make_log(log_id, day, logs, eid=5, name="Bench Press"):
    return ExerciseLog(
        **{"_id": log_id, "mydate": day, "eid": eid, "ename": name, "logs": logs}
    )


def test_old_weights_converted_from_lbs_and_raw_kept():
    plan = build_plan([make_log(1, "2025-12-31", "100x5")])
    s = plan.sets[0]
    assert s.weight_kg == 45.36
    assert s.weight_raw == 100.0


def test_2026_weights_not_converted():
    plan = build_plan([make_log(1, "2026-01-01", "80x8")])
    assert plan.sets[0].weight_kg == 80.0


def test_set_numbers_start_at_one_per_log():
    plan = build_plan([make_log(1, "2026-02-01", "80x8,80x7,75x6")])
    assert [s.set_number for s in plan.sets] == [1, 2, 3]
    assert {s.exercise_log_id for s in plan.sets} == {1}


def test_exercises_are_deduplicated():
    plan = build_plan([
        make_log(1, "2026-02-01", "80x8"),
        make_log(2, "2026-02-08", "82.5x8"),
    ])
    assert plan.exercises == {5: "Bench Press"}
    assert len(plan.logs) == 2


def test_bad_token_is_reported_and_loading_continues():
    plan = build_plan([
        make_log(1, "2026-02-01", "80xabc"),
        make_log(2, "2026-02-08", "82.5x8"),
    ])
    assert len(plan.errors) == 1
    assert "1" in plan.errors[0]          # mentions the log id
    assert len(plan.sets) == 1            # log 2 still loaded


def test_log_with_only_placeholder_sets_is_kept_with_no_sets():
    plan = build_plan([make_log(1, "2026-02-01", "0x0,0x0")])
    assert len(plan.logs) == 1
    assert plan.sets == []

def test_exercise_names_have_extra_spaces_removed():
    log = make_log(1, "2026-02-01", "80x8", eid=294, name="  Barbell  Ab Rollout ")
    plan = build_plan([log])
    assert plan.exercises == {294: "Barbell Ab Rollout"}