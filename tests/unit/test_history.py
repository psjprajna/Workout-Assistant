"""Unit tests: shrink_old_tool_results keeps call/result pairs and drops sets from old turns."""

from typing import cast

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from workout_agent.sessions.history import shrink_old_tool_results

SESSION = {
    "status": "ok",
    "exercise_id": 2,
    "exercise_name": "Barbell Bench Press",
    "performed_on": "2026-02-08",
    "exercise_log_id": 10,
    "sets": [{"set_number": 1, "weight_kg": "80.00", "reps": 8}],
}
FOUND = {"status": "ok", "matches": [{"exercise_id": 2, "name": "Barbell Bench Press"}]}


def _turn(n: int) -> list:
    """One turn: question, find + last session calls, their results, the answer."""
    return [
        ModelRequest(parts=[UserPromptPart(content=f"question {n}")]),
        ModelResponse(parts=[ToolCallPart("find_exercise_tool", {"query": "bench"}, f"f{n}")]),
        ModelRequest(parts=[ToolReturnPart("find_exercise_tool", dict(FOUND), f"f{n}")]),
        ModelResponse(parts=[ToolCallPart("get_last_session_tool", {"exercise_id": 2}, f"g{n}")]),
        ModelRequest(parts=[ToolReturnPart("get_last_session_tool", dict(SESSION), f"g{n}")]),
        ModelResponse(parts=[TextPart(content=f"answer {n}")]),
    ]


def _returns(messages) -> list[ToolReturnPart]:
    return [
        p for m in messages if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, ToolReturnPart)
    ]


def _calls(messages) -> list[ToolCallPart]:
    return [
        p for m in messages if isinstance(m, ModelResponse)
        for p in m.parts if isinstance(p, ToolCallPart)
    ]


def test_old_turns_lose_sets_and_recent_turns_keep_them():
    messages = _turn(1) + _turn(2) + _turn(3)
    shrunk = shrink_old_tool_results(messages, keep_full_turns=2)

    sessions = [r.content for r in _returns(shrunk) if r.tool_name == "get_last_session_tool"]
    assert sessions[0] == {
        "status": "ok",
        "exercise_id": 2,
        "exercise_name": "Barbell Bench Press",
        "performed_on": "2026-02-08",
    }
    assert sessions[1] == SESSION
    assert sessions[2] == SESSION


def test_find_exercise_keeps_status_and_matches():
    shrunk = shrink_old_tool_results(_turn(1) + _turn(2), keep_full_turns=1)
    assert _returns(shrunk)[0].content == FOUND


def test_every_call_keeps_its_result():
    messages = _turn(1) + _turn(2) + _turn(3)
    shrunk = shrink_old_tool_results(messages, keep_full_turns=0)

    assert len(shrunk) == len(messages)
    assert [c.tool_call_id for c in _calls(shrunk)] == [
        r.tool_call_id for r in _returns(shrunk)
    ]
    assert all("sets" not in cast(dict, r.content) for r in _returns(shrunk))


def test_input_is_not_changed():
    messages = _turn(1) + _turn(2)
    shrink_old_tool_results(messages, keep_full_turns=0)
    assert _returns(messages)[1].content == SESSION


def test_fewer_turns_than_kept_changes_nothing():
    messages = _turn(1)
    assert shrink_old_tool_results(messages, keep_full_turns=2) == messages
