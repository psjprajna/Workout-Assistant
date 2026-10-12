"""Integration tests: find_exercise returns ok / ambiguous / no_rows against a real database."""

from workout_agent.embeddings.search import MIN_SCORE, ScoredExercise
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


def fake_search(hits: list[ScoredExercise]):
    """A stand-in for search_exercises. Returns (search, calls): calls records each query."""
    calls: list[str] = []

    async def search(query: str) -> list[ScoredExercise]:
        calls.append(query)
        return hits

    return search, calls


ABOVE = MIN_SCORE + 0.1
BELOW = MIN_SCORE - 0.1


async def test_word_match_wins_and_meaning_search_is_not_called(conn):
    await _seed(conn)
    search, calls = fake_search([ScoredExercise(112, "Barbell Curl", ABOVE)])

    result = await find_exercise(conn, "bench press", search)

    assert result.status == "ambiguous"
    assert {m.exercise_id for m in result.matches} == {2, 4, 8}
    assert calls == []


async def test_no_word_match_offers_up_to_three_close_meanings_in_score_order(conn):
    await _seed(conn)
    search, calls = fake_search(
        [
            ScoredExercise(2, "Barbell Bench Press", ABOVE + 0.03),
            ScoredExercise(8, "Barbell Incline Bench Press", ABOVE + 0.02),
            ScoredExercise(4, "Barbell Close Grip Bench Press", ABOVE + 0.01),
            ScoredExercise(112, "Barbell Curl", ABOVE),
        ]
    )

    result = await find_exercise(conn, "chest press", search)

    assert result.status == "ambiguous"
    assert [m.exercise_id for m in result.matches] == [2, 8, 4]
    assert calls == ["chest press"]


async def test_one_close_meaning_is_still_asked_not_assumed(conn):
    await _seed(conn)
    search, calls = fake_search(
        [
            ScoredExercise(2, "Barbell Bench Press", ABOVE),
            ScoredExercise(112, "Barbell Curl", BELOW),
        ]
    )

    result = await find_exercise(conn, "chest press", search)

    assert result.status == "ambiguous"
    assert [m.exercise_id for m in result.matches] == [2]


async def test_nothing_close_in_meaning_is_no_rows(conn):
    await _seed(conn)
    search, calls = fake_search([ScoredExercise(112, "Barbell Curl", BELOW)])

    result = await find_exercise(conn, "zzz", search)

    assert result.status == "no_rows"
    assert result.matches == []


async def test_blank_query_never_searches(conn):
    await _seed(conn)
    search, calls = fake_search([ScoredExercise(112, "Barbell Curl", ABOVE)])

    result = await find_exercise(conn, "  ", search)

    assert result.status == "no_rows"
    assert calls == []


# --- staged search: whole words, shorthand, muscle words, spelling, not-an-exercise ---


async def _seed_more(conn):
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [
            (3, "Barbell Bent-Over Row"),
            (88, "Dumbbell Bent-Over Row"),
            (12, "Barbell Squat"),
            (47, "Push Up"),
            (52, "T Bar Row"),
            (627, "Modified Push-Up to Forearms"),
            (119, "Barbell Palms Up Wrist Curl"),
            (32, "Dumbbell Lateral Raise"),
            (86, "Wide Grip Lat Pulldown"),
        ],
    )


async def test_letters_inside_a_word_do_not_match(conn):
    await _seed_more(conn)
    # "bb" is shorthand for barbell, and never matches the "bb" inside "Dumbbell".
    result = await find_exercise(conn, "bb row")
    assert result.status == "ok"
    assert [m.exercise_id for m in result.matches] == [3]


async def test_db_means_dumbbell(conn):
    await _seed_more(conn)
    result = await find_exercise(conn, "db row")
    assert [m.exercise_id for m in result.matches] == [88]


async def test_plural_and_hyphen_still_match_whole_words(conn):
    await _seed_more(conn)
    assert [m.exercise_id for m in (await find_exercise(conn, "squats")).matches] == [12]
    assert [m.exercise_id for m in (await find_exercise(conn, "push ups")).matches] == [47, 627]
    assert [m.exercise_id for m in (await find_exercise(conn, "t-bar row")).matches] == [52]


async def test_lat_does_not_match_lateral(conn):
    await _seed_more(conn)
    result = await find_exercise(conn, "lat pulldown")
    assert [m.exercise_id for m in result.matches] == [86]


async def test_a_muscle_word_inside_a_name_is_asked_not_assumed(conn):
    await _seed_more(conn)
    search, _ = fake_search([ScoredExercise(119, "Barbell Palms Up Wrist Curl", ABOVE)])

    result = await find_exercise(conn, "forearms", search)

    assert result.status == "ambiguous"
    assert [m.exercise_id for m in result.matches] == [627, 119]


async def test_misspelt_word_is_fixed_and_offered(conn):
    await _seed_more(conn)
    result = await find_exercise(conn, "dumbell rows")
    assert result.status == "ambiguous"
    assert [m.exercise_id for m in result.matches] == [88]


async def test_a_day_or_workout_is_not_an_exercise(conn):
    await _seed_more(conn)
    search, calls = fake_search([ScoredExercise(47, "Push Up", ABOVE)])

    result = await find_exercise(conn, "push day", search)

    assert result.status == "no_rows"
    assert calls == []


async def test_regex_characters_in_the_query_are_plain_text(conn):
    await _seed_more(conn)
    result = await find_exercise(conn, "row (.*")
    assert result.status == "no_rows"
