"""Entry point for the server: uv run uvicorn workout_agent.api.main:app --reload

Builds the app with the rule-based router, the agent, and the database and Qdrant URLs
from Settings. The agent's find_exercise falls back to search by meaning via OpenRouter.
"""

from workout_agent.agents.workout_agent import build_agent
from workout_agent.api.app import create_app
from workout_agent.embeddings.openrouter import embed
from workout_agent.settings import Settings
from workout_agent.workflow.simple_router import simple_route

settings = Settings()


async def embedder(texts: list[str]) -> list[list[float]]:
    return await embed(settings, texts)


app = create_app(
    simple_route,
    settings.database_url,
    agent=build_agent(settings),
    qdrant_url=settings.qdrant_url,
    embedder=embedder,
)
