"""The workout agent: one model that decides which tool to call and which answer to give.

It can look up an exercise (find_exercise), fetch its last session (get_last_session),
and list the plan days with the suggested next one (workout_days). Once it chooses a
day, code fetches that day's exercises (tools/workout_day.planned_day): the long list
never goes to the model. It answers with one typed choice (agents/outputs.py), never with
text: code writes the reply from the tool results. Every tool result is copied into
deps.evidence. Guardrails (agents/guardrails.py) check the run's edges in code: the
safety screen stops a message before any model call; a tool call with an id no tool
returned, or a repeat, goes back to the model; a tool result outside its contract stops
the run; an answer about an id no tool returned this turn goes back to the model, so it
can't cite what it never fetched.
"""

from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

import asyncpg
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_harness.guardrails import InputGuardrail, OutputGuardrail, ToolGuardrail

from workout_agent.agents.guardrails import (
    answer_only_from_this_turn,
    safety_guard,
    tool_call_guard,
    tool_result_guard,
)
from workout_agent.agents.outputs import OUTPUT_TYPES, AgentOutput
from workout_agent.settings import Settings
from workout_agent.tools.find_exercise import Searcher, find_exercise
from workout_agent.tools.get_last_session import get_last_session
from workout_agent.tools.workout_day import workout_days

MODEL_NAME = "qwen/qwen3.5-35b-a3b"

INSTRUCTIONS = (
    "You answer questions about the user's workout log by choosing one final output. "
    "You never write the answer text: code writes it from the tool results. "
    "When the user names an exercise, call find_exercise with their words. "
    "If it returns 'ok', call get_last_session with that exercise_id, then choose "
    "ShowLastSession with the same exercise_id. "
    "If it returns 'ambiguous', choose AskWhich with the option ids in exactly the order "
    "returned, even when there is only one option; do not call get_last_session for them. "
    "If it returns 'no_rows', choose NotFound. "
    "If the user answers your last AskWhich with a number, that is the position in that "
    "list; with 'yes' to a single option, it is that option. Call get_last_session with "
    "that exercise_id, then choose ShowLastSession. "
    "To show any session, call get_last_session in this turn, even if you fetched it "
    "before. Never use an exercise id you did not get from a tool. "
    "When the user asks what to train today, tomorrow or next, or what is on a plan day "
    "(\"what's on Push B?\"), call workout_days, then choose ShowWorkoutDay with its "
    "suggested_day_id, or with the day_id of the day they named. "
    "For greetings and anything else, choose Help."
)


@dataclass
class AgentDeps:
    conn: asyncpg.Connection
    evidence: list[dict[str, Any]] = field(default_factory=list)  # every tool result
    search: Searcher | None = None  # search by meaning; None means word match only
    today: date = field(default_factory=date.today)  # breaks ties in the suggested day


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_plain(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, Decimal):
        return f"{value:.2f}"  # weights: exact text like "30.00", no float noise
    if isinstance(value, date):
        return value.isoformat()
    return value


def jsonable(result: Any) -> dict[str, Any]:
    """Tool result dataclass -> the dict the model reads and the evidence keeps."""
    return _plain(asdict(result))


def build_agent(settings: Settings) -> Agent[AgentDeps, AgentOutput]:
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = OpenRouterModel(
        MODEL_NAME,
        provider=OpenRouterProvider(api_key=settings.openrouter_api_key.get_secret_value()),
    )
    model_settings = OpenRouterModelSettings(
        temperature=0,
        # Your workout data goes to the model provider: refuse providers that train on it.
        openrouter_provider={"data_collection": "deny", "allow_fallbacks": False},
    )

    agent: Agent[AgentDeps, AgentOutput] = Agent(
        model,
        deps_type=AgentDeps,
        output_type=OUTPUT_TYPES,  # tool output mode: one output tool per choice
        instructions=INSTRUCTIONS,
        model_settings=model_settings,
        retries={"output": 2},  # room to fix an answer the output guardrail sends back
        capabilities=[
            InputGuardrail(guard=safety_guard),  # sequential: no model call when it trips
            ToolGuardrail(guard=tool_call_guard, result_guard=tool_result_guard),
            OutputGuardrail(guard=answer_only_from_this_turn),
        ],
    )

    @agent.tool
    async def find_exercise_tool(ctx: RunContext[AgentDeps], query: str) -> dict[str, Any]:
        """Find an exercise by name. Returns status ok / ambiguous / no_rows and matches.

        Use this whenever the user names an exercise.
        """
        data = jsonable(await find_exercise(ctx.deps.conn, query, ctx.deps.search))
        # The evidence also keeps the words searched for: replies quote them back.
        ctx.deps.evidence.append({**data, "query": query})
        return data

    @agent.tool
    async def get_last_session_tool(ctx: RunContext[AgentDeps], exercise_id: int) -> dict[str, Any]:
        """Get the most recent session (date, weights in kg, reps) for one exercise id.

        Only call this with an exercise_id returned by find_exercise.
        """
        data = jsonable(await get_last_session(ctx.deps.conn, exercise_id))
        ctx.deps.evidence.append(data)
        return data

    @agent.tool
    async def workout_days_tool(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
        """List the plan days, when each was last done, and suggested_day_id.

        suggested_day_id is the rotation day done longest ago: the next workout. Use this
        for "what should I train today / tomorrow / next?" and "what's on <day>?".
        """
        data = jsonable(await workout_days(ctx.deps.conn, ctx.deps.today))
        ctx.deps.evidence.append(data)
        return data

    return agent