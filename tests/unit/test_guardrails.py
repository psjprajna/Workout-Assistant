"""Unit tests: the agent's guardrails, called directly with a stand-in run context.

No model and no database: each guard is a plain function of (ctx, value).
"""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai_harness.guardrails import ToolBlocked, ToolCallInfo, ToolResultInfo

from workout_agent.agents.guardrails import (
    MAX_RESULT_CHARS,
    answer_only_from_this_turn,
    safety_guard,
    tool_call_guard,
    tool_result_guard,
)
from workout_agent.agents.outputs import Help, ShowLastSession
from workout_agent.workflow.safety import SAFETY_REPLY, SafetyRefused

FOUND = {"status": "ok", "matches": [{"exercise_id": 2, "name": "Barbell Bench Press"}]}
SESSION = {
    "status": "ok",
    "exercise_id": 2,
    "exercise_name": "Barbell Bench Press",
    "performed_on": "2026-02-08",
    "exercise_log_id": 10,
    "sets": [{"set_number": 1, "weight_kg": "80.00", "reps": 8}],
}


def _ctx(evidence: list[dict[str, Any]] = (), messages: list[Any] = ()) -> Any:  # type: ignore[assignment]
    return SimpleNamespace(deps=SimpleNamespace(evidence=list(evidence)), messages=list(messages))


def _get(exercise_id: int) -> ToolCallInfo:
    return ToolCallInfo(
        name="get_last_session_tool", args={"exercise_id": exercise_id}, tool_call_id="g"
    )


def _find(query: str) -> ToolCallInfo:
    return ToolCallInfo(name="find_exercise_tool", args={"query": query}, tool_call_id="f")


def _result(name: str, result: object) -> ToolResultInfo:
    return ToolResultInfo(name=name, args={}, tool_call_id="r", result=result)


# --- Input: the safety screen --------------------------------------------------------


def test_safety_guard_raises_the_fixed_refusal():
    with pytest.raises(SafetyRefused) as refused:
        safety_guard("my knee hurts, what should I squat?")
    assert refused.value.reply == SAFETY_REPLY


def test_safety_guard_allows_a_normal_question():
    assert safety_guard("last time I did bench press") is True


# --- Tool calls: only ids a tool returned, no repeats --------------------------------


def test_an_id_no_tool_returned_is_sent_back():
    verdict = tool_call_guard(_ctx(), _get(8))
    assert verdict.action == "retry"
    assert verdict.message == (
        "exercise_id 8 did not come from find_exercise; call find_exercise first"
    )


def test_an_id_from_this_turns_find_is_allowed():
    assert tool_call_guard(_ctx([{**FOUND, "query": "bench"}]), _get(2)).action == "allow"


def test_an_id_from_an_earlier_turn_is_allowed():
    history = [
        ModelRequest(parts=[UserPromptPart(content="bench?")]),
        ModelRequest(parts=[ToolReturnPart("find_exercise_tool", dict(FOUND), "f1")]),
    ]
    assert tool_call_guard(_ctx(messages=history), _get(2)).action == "allow"


def test_the_same_find_twice_in_a_turn_is_sent_back():
    ctx = _ctx([{**FOUND, "query": "Bench Press"}])
    verdict = tool_call_guard(ctx, _find("bench press"))
    assert verdict.action == "retry"
    assert verdict.message == "you already have this result this turn; use it"
    assert tool_call_guard(ctx, _find("incline bench")).action == "allow"


def test_the_same_session_twice_in_a_turn_is_sent_back():
    ctx = _ctx([{**FOUND, "query": "bench"}, SESSION])
    assert tool_call_guard(ctx, _get(2)).action == "retry"


# --- Tool results: the docs/tools.md contract ----------------------------------------


@pytest.mark.parametrize(
    ("name", "result"),
    [("find_exercise_tool", FOUND), ("get_last_session_tool", SESSION)],
)
def test_a_result_inside_the_contract_is_allowed(name, result):
    assert tool_result_guard(_ctx(), _result(name, result)) is True


@pytest.mark.parametrize(
    ("name", "result", "reason"),
    [
        ("find_exercise_tool", "ok", "result is not a dict"),
        ("find_exercise_tool", {**FOUND, "status": "conflict"}, "status 'conflict'"),
        ("get_last_session_tool", {**SESSION, "status": "ambiguous"}, "status 'ambiguous'"),
        ("get_last_session_tool", {**SESSION, "birth_date": "1990-01-01"}, "birth_date"),
        (
            "find_exercise_tool",
            {"status": "ok", "matches": [{"exercise_id": 2, "name": "x", "weight": 70}]},
            "weight",
        ),
        (
            "get_last_session_tool",
            {**SESSION, "sets": [{"set_number": 1, "weight_kg": "1", "reps": 1, "bmi": 22}]},
            "bmi",
        ),
        (
            "get_last_session_tool",
            {**SESSION, "exercise_name": "x" * MAX_RESULT_CHARS},
            "too large",
        ),
    ],
)
def test_a_result_outside_the_contract_stops_the_run(name, result, reason):
    with pytest.raises(ToolBlocked) as blocked:
        tool_result_guard(_ctx(), _result(name, result))
    assert blocked.value.tool_name == name
    assert reason in (blocked.value.reason or "")


# --- Output: answer only from this turn's tool results -------------------------------


def test_an_answer_about_an_unfetched_id_is_sent_back():
    verdict = answer_only_from_this_turn(_ctx(), ShowLastSession(exercise_id=2))
    assert verdict.action == "retry"
    assert "get_last_session was not called for exercise_id 2" in (verdict.message or "")


def test_an_answer_from_this_turn_is_allowed():
    assert answer_only_from_this_turn(_ctx([SESSION]), ShowLastSession(exercise_id=2)).action == (
        "allow"
    )
    assert answer_only_from_this_turn(_ctx(), Help()).action == "allow"
