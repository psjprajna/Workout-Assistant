from datetime import date
from decimal import Decimal

from workout_agent.tools.find_exercise import ExerciseMatch, FindResult
from workout_agent.tools.get_last_session import LastSessionResult, SetItem
from workout_agent.workflow.replies import (
    format_find_question,
    format_last_session,
    verify_reply,
)


def make_result(name="Barbell Bench Press", sets=None, status="ok"):
    if sets is None:
        sets = [
            SetItem(1, Decimal("80.00"), 8),
            SetItem(2, Decimal("80.00"), 7),
        ]
    return LastSessionResult(
        status=status,
        exercise_id=2,
        exercise_name=name,
        performed_on=date(2026, 2, 8),
        exercise_log_id=11,
        sets=sets,
    )


def test_last_session_reply_contains_name_date_and_sets():
    reply = format_last_session(make_result())
    assert "Barbell Bench Press" in reply
    assert "8 Feb 2026" in reply
    assert "80.00 kg x 8" in reply
    assert "80.00 kg x 7" in reply


def test_log_without_sets_says_so_and_shows_no_weights():
    reply = format_last_session(make_result(name="Walking", sets=[]))
    assert "Walking" in reply
    assert "no set details" in reply
    assert "kg" not in reply


def test_no_rows_reply_does_not_invent_anything():
    result = LastSessionResult(
        status="no_rows",
        exercise_id=99,
        exercise_name=None,
        performed_on=None,
        exercise_log_id=None,
    )
    reply = format_last_session(result)
    assert "can't find" in reply
    assert "kg" not in reply


def test_ambiguous_question_lists_numbered_options_in_order():
    result = FindResult(
        status="ambiguous",
        matches=[
            ExerciseMatch(2, "Barbell Bench Press"),
            ExerciseMatch(8, "Barbell Incline Bench Press"),
        ],
    )
    reply = format_find_question(result, "bench press")
    assert "1. Barbell Bench Press" in reply
    assert "2. Barbell Incline Bench Press" in reply


def test_not_found_question_repeats_the_query_and_says_not_in_data():
    result = FindResult(status="no_rows", matches=[])
    reply = format_find_question(result, "zzz")
    assert "zzz" in reply
    assert "not in your data" in reply


def test_verify_accepts_the_formatters_own_output():
    result = make_result()
    assert verify_reply(format_last_session(result), result)


def test_verify_rejects_a_weight_not_in_the_data():
    result = make_result()
    assert not verify_reply("You lifted 85.00 kg x 8 last time.", result)


def test_verify_rejects_wrong_reps():
    result = make_result()
    assert not verify_reply("Set 1: 80.00 kg x 9", result)


def test_verify_treats_80_and_80_00_as_the_same_number():
    result = make_result()
    assert verify_reply("Set 1: 80 kg x 8", result)


def test_verify_ignores_digits_inside_the_exercise_name():
    result = make_result(name="Cable Fly 45 Degree")
    assert verify_reply(format_last_session(result), result)


def test_reply_with_no_numbers_passes():
    assert verify_reply("Happy to help!", make_result())