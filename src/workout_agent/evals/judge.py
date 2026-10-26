"""The reply judge: two yes/no questions about the phrasing of one reply.

Offline only. It scores replies in evals (scripts/judge_results.py and the agent
experiment); it never runs in the request path and never sees the safety screen, the
evidence check, a calculation or verification. Why Sonnet 5.5: docs/amendments.md §25.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.typesafe import TypeSafeProvider

from workout_agent.settings import Settings

# Stronger than the agent's Qwen3.5-35B-A3B and from a different family, so it does not
# favour the agent's own phrasing.
DEFAULT_OPENROUTER = "anthropic/claude-sonnet-5.5"
QUESTIONS = ("answers_question", "clear_and_brief")

INSTRUCTIONS = """\
You grade one reply from a workout-log assistant. You see the user's message and the reply.
Answer two yes/no questions about the reply. Do not judge whether its numbers are true.

For answers_question: when the message is something the assistant cannot help with (a
greeting, an off-topic question), a reply that says what it can do answers it.

For clear_and_brief: a workout log is naturally a list. A numbered list of exercises, or
sets written out one by one, is the content the user asked for, not clutter. Say no for
padding, rambling, a cut-off reply, the same thing said twice, or internal error text
(an exception name or message) shown to the user.
"""


class Verdict(BaseModel):
    answers_question: bool = Field(
        description="The reply gives what was asked, or correctly asks what is needed."
    )
    clear_and_brief: bool = Field(
        description="Easy to read at a glance: no clutter, nothing repeated."
    )


def build_judge(spec: str, settings: Settings) -> Agent[None, Verdict]:
    """The judge named by spec: typesafe[:model] or openrouter[:slug]. Raises RuntimeError."""
    kind, _, name = spec.partition(":")
    model: Model
    if kind == "typesafe":
        if settings.typesafe_api_key is None or not settings.typesafe_api_key.get_secret_value():
            raise RuntimeError("TYPESAFE_API_KEY is not set (empty in .env)")
        provider = TypeSafeProvider(api_key=settings.typesafe_api_key.get_secret_value())
        model = TypeSafeModel(name or "jev-latest", provider=provider)
        return Agent(model, output_type=Verdict, instructions=INSTRUCTIONS)
    if kind == "openrouter":
        if settings.openrouter_api_key is None:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        key = settings.openrouter_api_key.get_secret_value()
        model = OpenRouterModel(
            name or DEFAULT_OPENROUTER, provider=OpenRouterProvider(api_key=key)
        )
        model_settings = OpenRouterModelSettings(
            temperature=0,
            openrouter_provider={"data_collection": "deny", "allow_fallbacks": False},
        )
        return Agent(
            model, output_type=Verdict, instructions=INSTRUCTIONS, model_settings=model_settings
        )
    raise RuntimeError(f"judge must be typesafe[:model] or openrouter[:slug], got {spec!r}")


async def judge_one(agent: Agent[None, Verdict], message: str, reply: str) -> dict[str, Any]:
    try:
        run = await agent.run(f"User message:\n{message}\n\nReply:\n{reply}")
    except Exception as e:  # one failed call is recorded; the rest still run
        return {"error": repr(e)}
    return run.output.model_dump()
