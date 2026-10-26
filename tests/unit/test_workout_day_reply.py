"""Unit tests: the workout-day reply is written from the tool results and passes the
number check; a choice no workout_days result backs is sent back."""

import pytest

from workout_agent.agents.outputs import ShowWorkoutDay
from workout_agent.workflow.replies import (
    RenderError,
    check_output,
    render,
    verify_agent_reply,
)

# As the evidence keeps them (agents/workout_agent.jsonable): text weights, ISO dates.
DAYS = {
    "status": "ok",
    "suggested_day_id": 3,
    "data_through": "2026-10-05",
    "days": [
        {"day_id": 1, "name": "Push A. Chest and shoulder", "in_rotation": True,
         "last_done": "2026-10-05"},
        {"day_id": 3, "name": "Legs A", "in_rotation": True, "last_done": "2026-09-23"},
        {"day_id": 4, "name": "Arms ", "in_rotation": False, "last_done": "2026-08-29"},
    ],
}
LEGS = {
    "status": "ok",
    "day_id": 3,
    "day_name": "Legs A",
    "exercises": [
        {"exercise_id": 216, "exercise_name": "Machine Hip Abduction", "timed": False,
         "target_reps": [30, 30, 30], "last_on": "2026-10-03",
         "last_sets": [{"set_number": 1, "weight_kg": "35.00", "reps": 30},
                       {"set_number": 2, "weight_kg": "40.00", "reps": 30},
                       {"set_number": 3, "weight_kg": "45.00", "reps": 30}]},
        {"exercise_id": 134, "exercise_name": "Dumbbell Step-Up", "timed": False,
         "target_reps": [10, 10, 10], "last_on": None, "last_sets": []},
        {"exercise_id": 631, "exercise_name": "Plank", "timed": True,
         "target_reps": [10, 10, 10], "last_on": "2026-10-05", "last_sets": []},
    ],
}
PUSH = {**LEGS, "day_id": 1, "day_name": "Push A. Chest and shoulder", "exercises": []}

LEGS_REPLY = """\
Your next workout is probably Legs A: of your rotation days, it's the one you did longest \
ago (last on 23 Sep 2026). This is from your log up to 5 Oct 2026; tell me if you've \
trained since.

1. Machine Hip Abduction: 3 sets of 30, 30, 30 reps; last time (3 Oct 2026): 35.00 kg x 30, \
40.00 kg x 30, 45.00 kg x 30
2. Dumbbell Step-Up: 3 sets of 10, 10, 10 reps; not logged yet
3. Plank: 3 sets, timed; last done 5 Oct 2026, no set details

Training a different day? Ask, for example, "what's on Push A?\""""


def test_the_suggested_day_says_why_and_how_fresh_the_log_is():
    assert render(ShowWorkoutDay(day_id=3), [DAYS, LEGS]) == LEGS_REPLY


def test_a_named_day_is_shown_without_the_suggestion():
    reply = render(ShowWorkoutDay(day_id=1), [DAYS, PUSH])
    assert reply.startswith("Push A, last done on 5 Oct 2026:")
    assert "No exercises are planned on Push A." in reply
    assert "probably" not in reply


def test_the_reply_passes_the_number_check():
    assert verify_agent_reply(LEGS_REPLY, [DAYS, LEGS])
    assert not verify_agent_reply(LEGS_REPLY.replace("45.00 kg", "47.50 kg"), [DAYS, LEGS])
    assert not verify_agent_reply(LEGS_REPLY.replace("30, 30, 30", "30, 30, 31"), [DAYS, LEGS])


@pytest.mark.parametrize(
    ("evidence", "problem"),
    [
        ([], "workout_days was not called in this turn"),
        ([DAYS], "day_id 9 is not one of the days workout_days returned: [1, 3, 4]"),
    ],
)
def test_a_day_no_workout_days_result_offered_is_sent_back(evidence, problem):
    day_id = 9 if evidence else 3
    assert problem in (check_output(ShowWorkoutDay(day_id=day_id), evidence) or "")


def test_a_chosen_day_is_allowed_before_its_exercises_are_fetched():
    # The output guardrail runs before code fetches the day, so only workout_days counts.
    assert check_output(ShowWorkoutDay(day_id=3), [DAYS]) is None
    with pytest.raises(RenderError):
        render(ShowWorkoutDay(day_id=3), [DAYS])  # but nothing can be written without it
