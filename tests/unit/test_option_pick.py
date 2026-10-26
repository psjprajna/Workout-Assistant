"""Unit tests: a bare number picks from the last AskWhich, in code."""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from workout_agent.workflow.option_pick import picked_exercise_id


def _turn(output: str, **args) -> list:
    """A turn that ended in the given final choice."""
    return [
        ModelRequest(parts=[UserPromptPart(content="question")]),
        ModelResponse(parts=[ToolCallPart(f"final_result_{output}", args, "c1")]),
        ModelRequest(parts=[ToolReturnPart(f"final_result_{output}", "ok", "c1")]),
    ]


ASKED = _turn("AskWhich", option_ids=[2, 8])


def test_a_number_picks_by_position_not_by_id():
    assert picked_exercise_id(ASKED, "2") == 8
    assert picked_exercise_id(ASKED, "1") == 2


def test_a_number_with_spaces_or_a_dot_still_picks():
    assert picked_exercise_id(ASKED, " 2. ") == 8


def test_out_of_range_zero_and_words_go_to_the_model():
    for text in ("3", "0", "yes", "the incline one", "2 please", "2.5"):
        assert picked_exercise_id(ASKED, text) is None


def test_no_pick_unless_the_last_turn_asked_which():
    assert picked_exercise_id([], "1") is None
    assert picked_exercise_id(_turn("ShowLastSession", exercise_id=2), "1") is None
    assert picked_exercise_id(ASKED + _turn("Help"), "1") is None
