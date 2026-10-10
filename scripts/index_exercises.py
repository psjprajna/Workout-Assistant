"""Embed every exercise name into Qdrant.

Usage: uv run python scripts/index_exercises.py
Needs OPENROUTER_API_KEY in .env and the Postgres and Qdrant containers running.
Prints the row count in Postgres and the point count in Qdrant; exits 1 if they differ.
Safe to run again: points are keyed by exercise_id, so nothing is duplicated.
"""

import asyncio
import sys

import asyncpg
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.index import COLLECTION, index_exercises
from workout_agent.embeddings.openrouter import MODEL_NAME, embed
from workout_agent.settings import Settings


async def main() -> int:
    settings = Settings()

    async def embedder(texts: list[str]) -> list[list[float]]:
        return await embed(settings, texts)

    conn = await asyncpg.connect(settings.database_url)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        points = await index_exercises(conn, qdrant, embedder)
        rows = await conn.fetchval("select count(*) from exercises")
        info = await qdrant.get_collection(COLLECTION) if points else None
    finally:
        await conn.close()
        await qdrant.close()

    print(f"model:          {MODEL_NAME}")
    if info is not None:
        print(f"vector size:    {info.config.params.vectors.size}")  # type: ignore[union-attr]
    print(f"db exercises:   {rows}")
    print(f"qdrant points:  {points}")
    return 0 if rows == points else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
