"""Unit tests: the judge evaluator scores phrasing per case from a scripted model.

No network: a FunctionModel stands in for the judge and answers by the reply text.
"""

import asyncio
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.evals.agent import judge_case
from workout_agent.evals.judge import Verdict, build_judge
from workout_agent.settings import Settings


def _judge(boom: str | None = None) -> Agent[None, Verdict]:
    """Says yes to both questions, except: "bad" fails aq, "ugly" fails cb, boom raises."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages[0].parts[-1].content)  # type: ignore[union-attr]
        if boom and boom in prompt:
            raise RuntimeError("judge down")
        args = {"answers_question": "bad" not in prompt, "clear_and_brief": "ugly" not in prompt}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return Agent(FunctionModel(model), output_type=Verdict)


def _turn(reply: str, kind: str = "show_last") -> dict[str, Any]:
    return {"message": "hi", "reply": reply, "kind": kind}


def _score(output: list[dict[str, Any]], judge: Agent[None, Verdict]) -> dict[str, Any]:
    return {e.name: e for e in asyncio.run(judge_case(judge)(output=output))}


def test_scores_are_fractions_over_the_turns() -> None:
    scores = _score([_turn("fine"), _turn("bad"), _turn("ugly"), _turn("fine")], _judge())
    assert scores["answers_question"].value == 0.75
    assert scores["clear_and_brief"].value == 0.75
    assert "judge_errors" not in scores


def test_refusal_turns_are_skipped() -> None:
    scores = _score([_turn("fixed refusal bad", "refuse"), _turn("fine")], _judge())
    assert scores["answers_question"].value == 1.0


def test_a_refusal_only_case_has_no_scores() -> None:
    assert _score([_turn("no", "refuse")], _judge()) == {}


def test_a_failed_call_is_recorded_not_raised() -> None:
    scores = _score([_turn("fine"), _turn("boom"), _turn("bad")], _judge(boom="boom"))
    assert scores["judge_errors"].value == 1
    assert "judge down" in scores["judge_errors"].comment
    assert scores["answers_question"].value == 0.5  # over the two judged turns
    assert scores["clear_and_brief"].value == 1.0


def test_all_calls_failing_leaves_only_the_error_score() -> None:
    assert set(_score([_turn("boom")], _judge(boom="boom"))) == {"judge_errors"}


def test_build_judge_without_a_key_raises_runtime_error() -> None:
    settings = Settings(openrouter_api_key=None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_judge("openrouter", settings)
    with pytest.raises(RuntimeError, match="must be"):
        build_judge("nope", settings)
