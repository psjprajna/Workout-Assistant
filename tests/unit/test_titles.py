"""Unit tests: chat titles come from tool results, else from the first message."""

from workout_agent.sessions.titles import chat_title


def test_last_session_names_the_chat():
    evidence = [
        {"status": "ok", "matches": [{"exercise_id": 2, "name": "Barbell Bench Press"}]},
        {"status": "ok", "exercise_id": 2, "exercise_name": "Barbell Bench Press", "sets": []},
    ]
    assert chat_title("last time bench?", evidence) == (
        "Barbell Bench Press – last session",
        False,
    )


def test_found_exercise_names_the_chat():
    evidence = [{"status": "ok", "matches": [{"exercise_id": 112, "name": "Barbell Curl"}]}]
    assert chat_title("curls?", evidence) == ("Barbell Curl", False)


def test_ambiguous_or_missing_falls_back_to_the_message():
    evidence = [
        {"status": "ambiguous", "matches": [{"exercise_id": 2, "name": "A"}]},
        {"status": "no_rows", "exercise_id": 9, "exercise_name": None, "sets": []},
    ]
    assert chat_title("  last   time I did bench press? ", evidence) == (
        "last time I did bench press?",
        True,
    )


def test_long_message_is_trimmed():
    title, fallback = chat_title("word " * 30, [])
    assert fallback
    assert len(title) <= 40
    assert title.endswith("…")
