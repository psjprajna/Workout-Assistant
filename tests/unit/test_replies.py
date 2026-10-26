"""Unit tests: reply templates, and verify_reply / verify_agent_reply catching numbers
not in the data."""

from datetime import date
from decimal import Decimal
from typing import Literal

import pytest

from workout_agent.agents.outputs import AskWhich, Help, NotFound, ShowLastSession
from workout_agent.tools.find_exercise import ExerciseMatch, FindResult
from workout_agent.tools.get_last_session import LastSessionResult, SetItem
from workout_agent.workflow.replies import (
    HELP_TEXT,
    RenderError,
    check_output,
    format_find_question,
    format_last_session,
    render,
    verify_agent_reply,
    verify_reply,
)


def make_result(name="Barbell Bench Press", sets=None, status: Literal["ok", "no_rows"] = "ok"):
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

# --- verify_agent_reply: the agent's own wording against the tool results it saw ------

# Tool results as the agent stores them in deps.evidence (agents/workout_agent.jsonable).
SESSION = {
    "status": "ok",
    "exercise_id": 2,
    "exercise_name": "Barbell Bench Press",
    "performed_on": "2026-02-08",
    "exercise_log_id": 11,
    "sets": [
        {"set_number": 1, "weight_kg": "80.00", "reps": 8},
        {"set_number": 2, "weight_kg": "80.00", "reps": 7},
        {"set_number": 3, "weight_kg": "75.00", "reps": 10},
    ],
}
AMBIGUOUS = {
    "status": "ambiguous",
    "matches": [
        {"exercise_id": 2, "name": "Barbell Bench Press"},
        {"exercise_id": 8, "name": "Barbell Incline Bench Press"},
    ],
}


def test_agent_reply_with_numbers_from_the_tools_passes():
    reply = "On 8 Feb 2026 you did 3 sets: 80.00 kg x 8, 80.00 kg x 7, 75.00 kg x 10."
    assert verify_agent_reply(reply, [AMBIGUOUS, SESSION])


def test_agent_reply_with_an_invented_weight_fails():
    assert not verify_agent_reply("You lifted 85 kg for 8 reps.", [SESSION])


def test_agent_reply_80_and_80_00_are_the_same_number():
    assert verify_agent_reply("You lifted 80 kg for 8 reps.", [SESSION])


def test_agent_reply_iso_date_passes():
    assert verify_agent_reply("Last done on 2026-02-08.", [SESSION])


def test_agent_reply_wrong_date_fails():
    assert not verify_agent_reply("Last done on 9 Feb 2026.", [SESSION])


def test_agent_reply_ignores_digits_inside_an_exercise_name():
    session = {**SESSION, "exercise_name": "Cable Fly 45 Degree"}
    assert verify_agent_reply("Cable Fly 45 Degree: 80 kg x 8", [session])


def test_agent_numbered_options_pass_up_to_the_option_count():
    reply = "Which one?\n1. Barbell Bench Press\n2. Barbell Incline Bench Press"
    assert verify_agent_reply(reply, [AMBIGUOUS])
    assert not verify_agent_reply(reply + "\n3. Something else", [AMBIGUOUS])


def test_agent_reply_with_no_tool_results_may_have_no_numbers():
    assert verify_agent_reply("I can look up your last session of an exercise.", [])
    assert not verify_agent_reply("You usually do 5 sets.", [])


# --- render: the agent's typed choice -> reply text, from this turn's tool results -----

FOUND_AMBIGUOUS = {**AMBIGUOUS, "query": "bench press"}
FOUND_NOTHING = {"status": "no_rows", "matches": [], "query": "zzz"}


def test_render_show_last_session_uses_the_rule_path_template():
    reply = render(ShowLastSession(exercise_id=2), [FOUND_AMBIGUOUS, SESSION])
    assert reply.splitlines() == [
        "Last time you did Barbell Bench Press was 8 Feb 2026:",
        "- Set 1: 80.00 kg x 8",
        "- Set 2: 80.00 kg x 7",
        "- Set 3: 75.00 kg x 10",
    ]


def test_render_ask_which_lists_the_options_in_the_order_chosen():
    reply = render(AskWhich(option_ids=[8, 2]), [FOUND_AMBIGUOUS])
    assert reply.splitlines() == [
        "I found more than one match for 'bench press'. Which one do you mean?",
        "1. Barbell Incline Bench Press",
        "2. Barbell Bench Press",
    ]


def test_render_ask_which_with_one_option_asks_did_you_mean():
    reply = render(AskWhich(option_ids=[8]), [FOUND_AMBIGUOUS])
    assert reply == (
        "I couldn't find 'bench press' exactly. Did you mean Barbell Incline Bench Press?"
    )


def test_render_not_found_and_help():
    assert render(NotFound(), [FOUND_NOTHING]) == "'zzz' is not in your data."
    assert render(Help(), []) == HELP_TEXT


@pytest.mark.parametrize(
    ("output", "evidence"),
    [
        (ShowLastSession(exercise_id=8), [SESSION]),  # fetched 2, not 8
        (ShowLastSession(exercise_id=2), [FOUND_AMBIGUOUS]),  # found, never fetched
        (AskWhich(option_ids=[2, 99]), [FOUND_AMBIGUOUS]),  # 99 was not offered
        (AskWhich(option_ids=[2, 2]), [FOUND_AMBIGUOUS]),
        (AskWhich(option_ids=[2]), [SESSION]),  # no ambiguous find this turn
        (NotFound(), [FOUND_AMBIGUOUS]),  # the search did find things
    ],
)
def test_an_output_that_cites_what_no_tool_returned_is_an_error(output, evidence):
    assert check_output(output, evidence) is not None
    with pytest.raises(RenderError):
        render(output, evidence)


def test_a_rendered_reply_always_passes_the_number_check():
    for output, evidence in [
        (ShowLastSession(exercise_id=2), [SESSION]),
        (AskWhich(option_ids=[2, 8]), [FOUND_AMBIGUOUS]),
        (NotFound(), [{**FOUND_NOTHING, "query": "squat 5x5"}]),  # digits in the query
    ]:
        assert verify_agent_reply(render(output, evidence), evidence)
