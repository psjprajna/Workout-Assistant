"""Search exercise names by meaning in Qdrant, for find_exercise's fallback (docs/tools.md).

Returns raw scores, best first, with no cutoff: find_exercise applies MIN_SCORE, and
scripts/score_queries.py prints the raw scores so the cutoff can be checked.
"""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.index import COLLECTION, Embedder

# Cosine score a meaning match needs before find_exercise offers it as "did you mean…?".
# Measured with scripts/score_queries.py on the real 154 exercises (text-embedding-3-small):
#   "chest press"  0.556 Barbell Bench Press, 0.555 Cable Mid Chest Crossover,
#                  0.547 Machine Incline Chest Press
#   "bicep curl"   0.769 Cable Bicep Curl, 0.753 Dumbbell Bicep Curl, 0.730 Barbell Curl
#   "zzz"          0.281 Walking, 0.268 Decline Crunch, 0.261 Flutter Kick
#   ("asdfgh" tops out at 0.267; "leg day" gives 0.443 Leg Extension, 0.434 Leg Press.)
# Noise stays under 0.29 and real matches start above 0.44, so the cutoff sits between them.
MIN_SCORE = 0.40


@dataclass
class ScoredExercise:
    exercise_id: int
    name: str
    score: float


async def search_exercises(
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    query: str,
    limit: int = 3,
    collection: str = COLLECTION,
) -> list[ScoredExercise]:
    """Return up to `limit` exercises closest in meaning to `query`, best first."""
    clean_query = query.strip()
    if not clean_query:
        return []
    if not await qdrant.collection_exists(collection):
        return []

    [vector] = await embedder([clean_query])
    response = await qdrant.query_points(
        collection_name=collection, query=vector, limit=limit, with_payload=True
    )
    return [
        ScoredExercise(
            exercise_id=point.payload["exercise_id"],
            name=point.payload["name"],
            score=point.score,
        )
        for point in response.points
        if point.payload
    ]
