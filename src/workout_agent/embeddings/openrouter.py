"""Text embeddings via OpenRouter, for exercise-name matching (docs/amendments.md §24).

Returns plain vectors, not the tool ToolResult envelope: this is an internal helper a
future tool (find_exercise, backed by Qdrant) will call, not a tool the model sees.
"""

import httpx

from workout_agent.settings import Settings

# Chosen over text-embedding-3-small on the search eval (docs/amendments.md §24): on its
# own Qdrant ranks a right answer first for 26/28 held-out phrases (small: 23/28).
MODEL_NAME = "openai/text-embedding-3-large"


def collection_for(model: str, base: str) -> str:
    """The production collection for MODEL_NAME; a separate one to try another model."""
    if model == MODEL_NAME:
        return base
    return f"{base}__{model.split('/')[-1]}"

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

# A batch of 256 texts can take longer than httpx's 5-second default.
TIMEOUT_SECONDS = 60.0


async def embed(
    settings: Settings, texts: list[str], model: str = MODEL_NAME
) -> list[list[float]]:
    """Embed each text, returning one vector per text in the same order as `texts`."""
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(
            EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
            },
            json={
                "model": model,
                "input": texts,
                # Your exercise names go to the model provider: refuse providers that train on it.
                "provider": {"data_collection": "deny"},
            },
        )
        response.raise_for_status()

    data = response.json()["data"]
    return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]
