from datetime import date
from decimal import Decimal

from workout_agent.tools.get_last_session import get_last_session


async def _seed(conn):
    await conn.execute(
        "insert into exercises (exercise_id, name) values (2, 'Barbell Bench Press')"
    )
    await conn.executemany(
        "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
        "values ($1, $2, $3)",
        [
            (10, 2, date(2026, 2, 1)),
            (11, 2, date(2026, 2, 8)),
        ],
    )
    await conn.executemany(
        "insert into sets (exercise_log_id, set_number, weight_kg, weight_raw, reps) "
        "values ($1, $2, $3, $4, $5)",
        [
            (10, 1, Decimal("75.00"), Decimal("75.00"), 8),
            (11, 2, Decimal("80.00"), Decimal("80.00"), 7),
            (11, 1, Decimal("80.00"), Decimal("80.00"), 8),
        ],
    )


async def test_returns_latest_log_sets_in_order(conn):
    await _seed(conn)
    result = await get_last_session(conn, 2)
    assert result.status == "ok"
    assert result.exercise_name == "Barbell Bench Press"
    assert result.performed_on == date(2026, 2, 8)
    assert [s.set_number for s in result.sets] == [1, 2]
    assert [s.reps for s in result.sets] == [8, 7]


async def test_weights_are_exact_decimals(conn):
    await _seed(conn)
    result = await get_last_session(conn, 2)
    assert result.sets[0].weight_kg == Decimal("80.00")


async def test_exercise_with_no_logs_is_no_rows(conn):
    await _seed(conn)
    await conn.execute(
        "insert into exercises (exercise_id, name) values (99, 'Never Done')"
    )
    result = await get_last_session(conn, 99)
    assert result.status == "no_rows"
    assert result.sets == []


async def test_unknown_exercise_id_is_no_rows(conn):
    await _seed(conn)
    result = await get_last_session(conn, 12345)
    assert result.status == "no_rows"


async def test_latest_log_without_sets_is_ok_with_empty_sets(conn):
    await _seed(conn)
    await conn.execute(
        "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
        "values (12, 2, '2026-03-01')"
    )
    result = await get_last_session(conn, 2)
    assert result.status == "ok"
    assert result.performed_on == date(2026, 3, 1)
    assert result.sets == []


async def test_same_day_tie_uses_highest_log_id(conn):
    await _seed(conn)
    await conn.execute(
        "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
        "values (13, 2, '2026-02-08')"
    )
    await conn.execute(
        "insert into sets (exercise_log_id, set_number, weight_kg, weight_raw, reps) "
        "values (13, 1, 85.00, 85.00, 5)"
    )
    result = await get_last_session(conn, 2)
    assert result.exercise_log_id == 13
    assert result.sets[0].reps == 5