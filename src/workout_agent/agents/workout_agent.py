"""The workout agent: one model that decides which tool to call.

It can look up an exercise (find_exercise) and fetch its last session
(get_last_session). Every tool result is copied into deps.evidence, so we can check
afterwards that the reply only contains numbers the tools returned.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import asyncpg
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from workout_agent.settings import Settings
from workout_agent.tools.find_exercise import Searcher, find_exercise
from workout_agent.tools.get_last_session import get_last_session

MODEL_NAME = "qwen/qwen3.5-35b-a3b"

INSTRUCTIONS = (
    "You answer questions about the user's workout log. "
    "Always call find_exercise first; never use an exercise id you did not get from a tool. "
    "Only state facts returned by the tools. Never invent or assume numbers. "
    "If find_exercise returns 'ambiguous', list the exact options and ask which one. "
    "If a tool returns 'no_rows', say that item is not in the data. "
    "Keep answers to a couple of short lines."
    "When find_exercise returns 'ambiguous', show the options as a numbered list "
    "(1., 2., 3., ...) in exactly the order the tool returned them. "
    "If the user then replies with a number, that number is the position in that list; "
    "call get_last_session with that option's exercise_id. "
    "An 'ambiguous' result can also be close matches by meaning when no name matched "
    "the words: offer them the same way, as 'Did you mean' and the numbered list. "
    "'ambiguous' with a single option is still a question: ask 'Did you mean ...?' "
    "and wait; never call get_last_session for it until the user says yes. "
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


def build_agent(settings: Settings) -> Agent[AgentDeps, str]:
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

    agent = Agent(
        model,
        deps_type=AgentDeps,
        instructions=INSTRUCTIONS,
        model_settings=model_settings,
    )

    @agent.tool
    async def find_exercise_tool(ctx: RunContext[AgentDeps], query: str) -> dict[str, Any]:
        """Find an exercise by name. Returns status ok / ambiguous / no_rows and matches.

        Use this whenever the user names an exercise.
        """
        data = _jsonable(await find_exercise(ctx.deps.conn, query, ctx.deps.search))
        ctx.deps.evidence.append(data)
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