from workout_agent.importing.loader import build_plan, load_plan
from workout_agent.importing.parsers import ExerciseLog


def make_log(log_id, day, logs):
    return ExerciseLog(
        **{"_id": log_id, "mydate": day, "eid": 5, "ename": "Bench Press", "logs": logs}
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
