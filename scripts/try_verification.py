"""Show the number check on the agent path, on your real data, with no model calls.

Usage: uv run python scripts/try_verification.py ["barbell bench press"]
Two scripted models fetch the same last session through the real tools. One quotes the
first set's weight as the tool returned it; the other adds 5 kg. The first reply is
shown as verified, the second is refused. Everything runs in a transaction that is
rolled back, so no chat is saved.
"""

import asyncio
import sys
from decimal import Decimal
from typing import Any, cast

import asyncpg
from pydantic import SecretStr
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from workout_agent.agents.workout_agent import build_agent
from workout_agent.settings import Settings
from workout_agent.workflow.agent_chat import stream_agent_message
from workout_agent.workflow.events import Final, ToolEnd, ToolStart


def scripted(query: str, extra_kg: Decimal) -> FunctionModel:
    """find_exercise, get_last_session for the first match, then quote set 1 (+ extra_kg)."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("find_exercise_tool", {"query": query})])
        result = cast(dict[str, Any], returns[0].content)
        if returns[0].tool_name == "find_exercise_tool":
            if not result["matches"]:
                return ModelResponse(parts=[TextPart(f"'{query}' is not in your data.")])
            exercise_id = result["matches"][0]["exercise_id"]
            call = ToolCallPart("get_last_session_tool", {"exercise_id": exercise_id})
            return ModelResponse(parts=[call])
        if not result["sets"]:
            return ModelResponse(parts=[TextPart("No sets were logged.")])
        first = result["sets"][0]
        weight = Decimal(first["weight_kg"]) + extra_kg
        text = (
            f"Last time you did {result['exercise_name']} on {result['performed_on']}: "
            f"{weight} kg x {first['reps']}."
        )
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(model)


async def ask(conn: asyncpg.Connection, label: str, model: FunctionModel) -> None:
    agent = build_agent(Settings(openrouter_api_key=SecretStr("unused")))
    print(f"--- {label}")
    with agent.override(model=model):
        async for event in stream_agent_message(conn, f"verify-demo-{label}", "last time?", agent):
            if isinstance(event, ToolStart):
                print(f"  step   {event.label}")
            elif isinstance(event, ToolEnd):
                print(f"  done   {event.tool}: {event.status}")
            elif isinstance(event, Final):
                print(f"  reply  {event.reply}")
                mark = "✓ verified" if event.verified else "✗ not verified"
                print(f"  check  {mark} against {event.checked} tool results")
    print()


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "barbell bench press"
    conn = await asyncpg.connect(Settings().database_url)
    try:
        tx = conn.transaction()
        await tx.start()
        try:
            await ask(conn, "honest", scripted(query, Decimal("0")))
            await ask(conn, "invents 5 kg", scripted(query, Decimal("5")))
        finally:
            await tx.rollback()  # leave no demo chats behind
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
