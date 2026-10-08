"""Unit tests: the rule-based router's decisions for typical messages."""

from workout_agent.workflow.simple_router import simple_route


async def test_last_time_question_extracts_the_exercise():
    d = await simple_route("last time I did bench press?", False)
    assert d.intent == "last_session"
    assert d.exercise_text == "bench press"


async def test_number_with_pending_question_is_a_pick():
    d = await simple_route("2", True)
    assert d.intent == "pick_option"
    assert d.option_number == 2


async def test_number_without_pending_question_is_unknown():
    d = await simple_route("2", False)
    assert d.intent == "unknown"


async def test_other_text_is_unknown():
    d = await simple_route("what's the weather?", False)
    assert d.intent == "unknown"