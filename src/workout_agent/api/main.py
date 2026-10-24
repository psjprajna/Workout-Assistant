"""Entry point for the server: uv run uvicorn workout_agent.api.main:app --reload

Builds the app with the rule-based router, the agent, and the database and Qdrant URLs
from Settings. The agent's find_exercise falls back to search by meaning via OpenRouter.
Langfuse tracing is on when its keys are set.
"""

import asyncpg
from qdrant_client import AsyncQdrantClient

from workout_agent.agents.workout_agent import build_agent
from workout_agent.api.app import create_app
from workout_agent.embeddings.openrouter import embed
from workout_agent.importing.refresh import RefreshResult, refresh_from_jefit
from workout_agent.observability import setup_tracing
from workout_agent.settings import Settings
from workout_agent.workflow.simple_router import simple_route

settings = Settings()
setup_tracing(settings)  # traces to Langfuse when its keys are in .env


async def embedder(texts: list[str]) -> list[list[float]]:
    return await embed(settings, texts)


async def jefit_refresh(
    conn: asyncpg.Connection, qdrant: AsyncQdrantClient | None
) -> RefreshResult:
    """The first chat message of the day: download Jefit's export, import it, re-index."""
    return await refresh_from_jefit(settings, conn, qdrant, embedder)


app = create_app(
    simple_route,
    settings.database_url,
    agent=build_agent(settings),
    qdrant_url=settings.qdrant_url,
    embedder=embedder,
    history_turns=settings.history_turns,
    history_full_tool_turns=settings.history_full_tool_turns,
    jefit_refresh=jefit_refresh,
)
