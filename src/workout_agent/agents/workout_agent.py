"""The workout agent: one model that decides which tool to call and which answer to give.

It can look up an exercise (find_exercise) and fetch its last session
(get_last_session). It answers with one typed choice (agents/outputs.py), never with
text: code writes the reply from the tool results. Every tool result is copied into
deps.evidence. Guardrails (agents/guardrails.py) check the run's edges in code: the
safety screen stops a message before any model call; a tool call with an id no tool
returned, or a repeat, goes back to the model; a tool result outside its contract stops
the run; an answer about an id no tool returned this turn goes back to the model, so it
can't cite what it never fetched.
"""

from dataclasses import asdict, dataclass, field
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
    "For greetings and anything not about an exercise's last session, choose Help."
)


@dataclass
class AgentDeps:
    conn: asyncpg.Connection
    evidence: list[dict[str, Any]] = field(default_factory=list)  # every tool result
    search: Searcher | None = None  # search by meaning; None means word match only


def _jsonable(result: Any) -> dict[str, Any]:
    """Dataclass -> dict the model can read. Weights become strings like "30.00"."""
    data = asdict(result)
    for s in data.get("sets", []):
        s["weight_kg"] = f"{s['weight_kg']:.2f}"  # Decimal -> exact text, no float noise
    if data.get("performed_on") is not None:
        data["performed_on"] = data["performed_on"].isoformat()
    return data


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
        data = _jsonable(await find_exercise(ctx.deps.conn, query, ctx.deps.search))
        # The evidence also keeps the words searched for: replies quote them back.
        ctx.deps.evidence.append({**data, "query": query})
        return data

    @agent.tool
    async def get_last_session_tool(ctx: RunContext[AgentDeps], exercise_id: int) -> dict[str, Any]:
        """Get the most recent session (date, weights in kg, reps) for one exercise id.

        Only call this with an exercise_id returned by find_exercise.
        """
        data = _jsonable(await get_last_session(ctx.deps.conn, exercise_id))
        ctx.deps.evidence.append(data)
        return data

    return agent