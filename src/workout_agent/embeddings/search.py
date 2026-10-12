"""Search exercises by meaning in Qdrant, for find_exercise's fallback (docs/tools.md).

Returns raw scores, best first, with no cutoff: find_exercise applies MIN_SCORE, and
scripts/sweep_search.py replays a saved eval run to choose it.
"""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.index import COLLECTION, Embedder

# Cosine score a meaning match needs before find_exercise offers it as "did you mean…?".
# Chosen with scripts/sweep_search.py on evals/results/search-2026-10-12-phase-d1-multi-vector
# (text-embedding-3-small, several points per exercise, 154 exercises):
#   noise phrases top out at 0.489 ("how are you" -> Good Morning; "zzz" 0.423 -> RDL)
#   right answers by meaning start at 0.417, 0.450, then 0.533, 0.541, 0.556 …
#   every cutoff from 0.50 to 0.70 passes 88/88 dev and 30/31 holdout with no false offer.
# 0.52 sits just above the noise and keeps the weaker right answers a new phrase may need.
MIN_SCORE = 0.52


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
    # Several points per exercise (name, other names, description): one group per
    # exercise, scored by its closest point.
    response = await qdrant.query_points_groups(
        collection_name=collection,
        query=vector,
        group_by="exercise_id",
        group_size=1,
        limit=limit,
        with_payload=True,
    )
    return [
        ScoredExercise(
            exercise_id=group.hits[0].payload["exercise_id"],
            name=group.hits[0].payload["name"],
            score=group.hits[0].score,
        )
        for group in response.groups
        if group.hits and group.hits[0].payload
    ]
