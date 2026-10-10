"""Text embeddings via OpenRouter, for exercise-name matching (docs/amendments.md §24).

Returns plain vectors, not the tool ToolResult envelope: this is an internal helper a
future tool (find_exercise, backed by Qdrant) will call, not a tool the model sees.
"""

import httpx

from workout_agent.settings import Settings

MODEL_NAME = "openai/text-embedding-3-small"

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


async def embed(settings: Settings, texts: list[str]) -> list[list[float]]:
    """Embed each text, returning one vector per text in the same order as `texts`."""
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
            },
            json={
                "model": MODEL_NAME,
                "input": texts,
                # Your exercise names go to the model provider: refuse providers that train on it.
                "provider": {"data_collection": "deny"},
            },
        )
        response.raise_for_status()

    data = response.json()["data"]
    return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]
