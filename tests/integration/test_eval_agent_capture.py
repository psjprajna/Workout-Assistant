"""run_case keeps what the tools returned, for the offline Ragas scorer.

A FunctionModel stands in for the language model, so there are no network calls.
"""

import json
from typing import Any, cast

from pydantic import SecretStr
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.agents.workout_agent import build_agent
from workout_agent.evals.agent import FIND, LAST, Turn, TurnResult, run_case
from workout_agent.settings import Settings

AGENT = build_agent(Settings(openrouter_api_key=SecretStr("test")))


def call(tool: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args)])


def find_then_fetch(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
    if not returns:
        return call(FIND, query="barbell bench press")
    if returns[0].tool_name == FIND:
        return call(LAST, exercise_id=2)
    return call("final_result_ShowLastSession", exercise_id=2)


async def test_a_fetch_keeps_both_tool_results_and_calls(conn, seeded):
    turn = Turn("bench?", "show_last", [FIND, LAST], 2)
    with AGENT.override(model=FunctionModel(find_then_fetch)):
        [result] = await run_case(conn, AGENT, "capture", [turn])

    assert [json.loads(c)["status"] for c in result.contexts] == ["ok", "ok"]
    assert result.tool_calls == [
        {"tool": FIND, "args": {"query": "barbell bench press"}},
        {"tool": LAST, "args": {"exercise_id": 2}},
    ]
    assert result.offered_ids == []


async def test_an_ambiguous_find_keeps_the_ids_it_offered(conn, seeded):
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return call(FIND, query="bench press")
        matches = cast(dict[str, Any], returns[0].content)["matches"]
        return call("final_result_AskWhich", option_ids=[m["exercise_id"] for m in matches])

    turn = Turn("last bench press?", "ask_which", [FIND])
    with AGENT.override(model=FunctionModel(model)):
        [result] = await run_case(conn, AGENT, "capture ask", [turn])

    offered = [m["exercise_id"] for m in json.loads(result.contexts[0])["matches"]]
    assert len(offered) > 1
    assert result.offered_ids == offered


def test_a_result_saved_before_these_fields_still_loads():
    old: dict[str, Any] = dict(
        case="c", message="m", expected_kind="help", expected_tools=[], kind="help", tools=[],
        fetched_ids=[], output=None, reply="r", verified=True, seconds=0.1,
        input_tokens=1, output_tokens=1,
    )  # fmt: skip
    result = TurnResult(**old)
    assert (result.contexts, result.tool_calls, result.offered_ids) == ([], [], [])
