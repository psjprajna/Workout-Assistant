"""Show the checks on the agent path, on your real data, with no model calls.

Usage: uv run python scripts/try_verification.py ["barbell bench press"]
The model only chooses a typed answer; code writes the words. Two scripted models:
one finds the exercise, fetches its last session and chooses ShowLastSession; the other
chooses ShowLastSession for an exercise it never fetched. The first reply is shown as
verified. The second is sent back to the model with the reason until the retries run
out, and nothing is shown or saved. Everything runs in a transaction that is rolled
back, so no chat is saved either way.
"""

import asyncio
import sys
from typing import Any, cast

import asyncpg
from pydantic import SecretStr
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.agents.workout_agent import build_agent
from workout_agent.settings import Settings
from workout_agent.workflow.agent_chat import stream_agent_message
from workout_agent.workflow.events import Final, ToolEnd, ToolStart


def _call(tool: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args)])


def honest(query: str) -> FunctionModel:
    """find_exercise, get_last_session for the first match, then ShowLastSession."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return _call("find_exercise_tool", query=query)
        result = cast(dict[str, Any], returns[0].content)
        if returns[0].tool_name == "find_exercise_tool":
            if not result["matches"]:
                return _call("final_result_NotFound")
            return _call("get_last_session_tool", exercise_id=result["matches"][0]["exercise_id"])
        return _call("final_result_ShowLastSession", exercise_id=result["exercise_id"])

    return FunctionModel(model)


def cites_unfetched(query: str) -> FunctionModel:
    """find_exercise, then ShowLastSession for an exercise id it never fetched."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = messages[-1].parts
        for p in parts:
            if isinstance(p, RetryPromptPart):
                print(f"  sent back  {p.content}")
        returns = [p for p in parts if isinstance(p, ToolReturnPart)]
        if not returns and not any(isinstance(p, RetryPromptPart) for p in parts):
            return _call("find_exercise_tool", query=query)
        return _call("final_result_ShowLastSession", exercise_id=999_999)

    return FunctionModel(model)


async def ask(conn: asyncpg.Connection, label: str, query: str, model: FunctionModel) -> None:
    agent = build_agent(Settings(openrouter_api_key=SecretStr("unused")))
    print(f"--- {label}")
    with agent.override(model=model):
        try:
            async for event in stream_agent_message(
                conn, f"verify-demo-{label}", f"last time I did {query}?", agent
            ):
                if isinstance(event, ToolStart):
                    print(f"  step   {event.label}")
                elif isinstance(event, ToolEnd):
                    print(f"  done   {event.tool}: {event.status}")
                elif isinstance(event, Final):
                    print("  reply  " + event.reply.replace("\n", "\n         "))
                    mark = "✓ verified" if event.verified else "✗ not verified"
                    print(f"  check  {mark} against {event.checked} tool results")
        except Exception as e:  # what the API turns into an error event
            print(f"  failed {type(e).__name__}: {e}")
            print("  (nothing shown, nothing saved)")
    print()


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "barbell bench press"
    conn = await asyncpg.connect(Settings().database_url)
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await ask(conn, "fetches first", query, honest(query))
            await ask(conn, "cites an unfetched session", query, cites_unfetched(query))
        finally:
            await tx.rollback()  # leave no demo chats behind
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
