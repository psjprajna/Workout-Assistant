"""Embed every exercise's name, other names and description into Qdrant.

Usage: uv run python scripts/index_exercises.py
Needs OPENROUTER_API_KEY in .env and the Postgres and Qdrant containers running; run
scripts/load_catalog.py first so other names and muscles are indexed too. Prints the
exercise count in Postgres and in Qdrant; exits 1 if any exercise has no point.
Safe to run again: point ids come from the text, so nothing is duplicated.
"""

import asyncio
import sys

import asyncpg
from qdrant_client import AsyncQdrantClient

from workout_agent.db.migrate import apply_migrations
from workout_agent.embeddings.cache import cached
from workout_agent.embeddings.index import COLLECTION, index_exercises
from workout_agent.embeddings.openrouter import MODEL_NAME, embed
from workout_agent.settings import Settings


async def main() -> int:
    settings = Settings()

    async def fetch(texts: list[str]) -> list[list[float]]:
        return await embed(settings, texts)

    conn = await asyncpg.connect(settings.database_url)
    await apply_migrations(conn)
    embedder = cached(conn, MODEL_NAME, fetch)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        count = await index_exercises(conn, qdrant, embedder)
        rows = await conn.fetchval("select count(*) from exercises")
        info = await qdrant.get_collection(COLLECTION) if count.points else None
    finally:
        await conn.close()
        await qdrant.close()

    print(f"model:          {MODEL_NAME}")
    if info is not None:
        print(f"vector size:    {info.config.params.vectors.size}")  # type: ignore[union-attr]
    print(f"db exercises:     {rows}")
    print(f"qdrant exercises: {count.exercises}")
    print(f"qdrant points:    {count.points}")
    return 0 if rows == count.exercises else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
