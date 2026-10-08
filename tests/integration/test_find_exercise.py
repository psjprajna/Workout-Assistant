"""Integration tests: find_exercise returns ok / ambiguous / no_rows against a real database."""

from workout_agent.tools.find_exercise import find_exercise


async def _seed(conn):
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [
            (2, "Barbell Bench Press"),
            (4, "Barbell Close Grip Bench Press"),
            (8, "Barbell Incline Bench Press"),
            (66, "Barbell Ab Rollout"),
            (294, "Barbell Ab Rollout"),
            (112, "Barbell Curl"),
        ],
    )


async def test_single_match_is_ok(conn):
    await _seed(conn)
    result = await find_exercise(conn, "curl")
    assert result.status == "ok"
    assert [m.exercise_id for m in result.matches] == [112]


async def test_several_matches_is_ambiguous(conn):
    await _seed(conn)
    result = await find_exercise(conn, "bench press")
    assert result.status == "ambiguous"
    assert {m.exercise_id for m in result.matches} == {2, 4, 8}


async def test_exact_name_wins_over_longer_names(conn):
    await _seed(conn)
    result = await find_exercise(conn, "BARBELL BENCH PRESS")
    assert result.status == "ok"
    assert result.matches[0].exercise_id == 2


async def test_same_name_twice_is_ambiguous(conn):
    await _seed(conn)
    result = await find_exercise(conn, "ab rollout")
    assert result.status == "ambiguous"
    assert {m.exercise_id for m in result.matches} == {66, 294}


async def test_word_order_does_not_matter(conn):
    await _seed(conn)
    result = await find_exercise(conn, "press incline")
    assert result.status == "ok"
    assert result.matches[0].exercise_id == 8


async def test_nothing_found_is_no_rows(conn):
    await _seed(conn)
    result = await find_exercise(conn, "zzz")
    assert result.status == "no_rows"
    assert result.matches == []

async def test_exact_name_shared_by_two_exercises_is_ambiguous(conn):
    await _seed(conn)
    result = await find_exercise(conn, "Barbell Ab Rollout")
    assert result.status == "ambiguous"
    assert {m.exercise_id for m in result.matches} == {66, 294}