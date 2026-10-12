"""Print the top meaning matches and their scores for a few test phrases.

Usage: uv run python scripts/score_queries.py [phrase ...]
Needs OPENROUTER_API_KEY in .env, Qdrant running, and scripts/index_exercises.py run first.
For choosing MIN_SCORE, prefer scripts/sweep_search.py on a saved eval run.
"""

import asyncio
import sys

from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.openrouter import embed
from workout_agent.embeddings.search import MIN_SCORE, search_exercises
from workout_agent.settings import Settings

DEFAULT_PHRASES = ["chest press", "bicep curl", "zzz"]


async def main(phrases: list[str]) -> int:
    settings = Settings()

    async def embedder(texts: list[str]) -> list[list[float]]:
        return await embed(settings, texts)

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        for phrase in phrases:
            print(f"{phrase!r}")
            for hit in await search_exercises(qdrant, embedder, phrase, limit=5):
                mark = "*" if hit.score >= MIN_SCORE else " "
                print(f"  {mark} {hit.score:.3f}  {hit.name}")
    finally:
        await qdrant.close()

    print(f"* = at or above MIN_SCORE ({MIN_SCORE})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or DEFAULT_PHRASES)))
