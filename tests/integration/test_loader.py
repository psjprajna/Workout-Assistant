from workout_agent.importing.loader import build_plan, load_plan
from workout_agent.importing.parsers import ExerciseLog


def make_log(log_id, day, logs, eid=5, name="Bench Press"):
    return ExerciseLog(
        **{"_id": log_id, "mydate": day, "eid": eid, "ename": name, "logs": logs}
    )


async def _counts(conn):
    return (
        await conn.fetchval("select count(*) from exercises"),
        await conn.fetchval("select count(*) from exercise_logs"),
        await conn.fetchval("select count(*) from sets"),
    )


async def test_load_twice_does_not_duplicate(conn):
    plan = build_plan([
        make_log(1, "2026-02-01", "80x8,80x7"),
        make_log(2, "2026-02-08", "82.5x8"),
    ])
    await load_plan(conn, plan)
    first = await _counts(conn)
    await load_plan(conn, plan)
    second = await _counts(conn)

    assert first == (1, 2, 3)
    assert second == first

def test_timed_exercise_keeps_log_but_gets_no_sets():
    log = make_log(1, "2026-02-01", "0x600", eid=321, name="Treadmill Running")
    plan = build_plan([log], timed_exercise_ids=frozenset({321}))
    assert len(plan.logs) == 1
    assert plan.sets == []
    assert plan.exercises == {321: "Treadmill Running"}


def test_other_exercises_unaffected_by_timed_list():
    plan = build_plan(
        [make_log(1, "2026-02-01", "80x8")],
        timed_exercise_ids=frozenset({321}),
    )
    assert len(plan.sets) == 1


def test_timed_exercise_bad_token_is_not_an_error():
    log = make_log(1, "2026-02-01", "0x80.0,0x600", eid=321, name="Treadmill Running")
    plan = build_plan([log], timed_exercise_ids=frozenset({321}))
    assert plan.errors == []