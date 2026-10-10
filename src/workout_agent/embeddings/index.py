"""Index every exercise name into Qdrant, for exercise-name matching (docs/tools.md).

One point per exercise, with the exercise_id as the point id, so running the indexer
again overwrites points instead of adding duplicates. The vector size comes from the
embedding model's actual output, never a hard-coded number.
"""

from collections.abc import Awaitable, Callable

import asyncpg
from qdrant_client import AsyncQdrantClient, models

COLLECTION = "exercises"

# How many names go to the embedding model in one request.
BATCH_SIZE = 256

# Takes texts, returns one vector per text in the same order. The script passes the
# OpenRouter embed(); tests pass a fake so they need no API key.
Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


async def index_exercises(
    conn: asyncpg.Connection,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    collection: str = COLLECTION,
) -> int:
    """Embed every row of `exercises` into `collection`; return the collection's point count."""
    rows = await conn.fetch("select exercise_id, name from exercises order by exercise_id")
    if not rows:
        return 0

    names = [row["name"] for row in rows]
    vectors: list[list[float]] = []
    for start in range(0, len(names), BATCH_SIZE):
        batch = names[start : start + BATCH_SIZE]
        batch_vectors = await embedder(batch)
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(batch_vectors)} vectors for {len(batch)} names"
            )
        vectors.extend(batch_vectors)

    size = len(vectors[0])
    await _ensure_collection(qdrant, collection, size)

    ids = [row["exercise_id"] for row in rows]
    await qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=exercise_id,
                vector=vector,
                payload={"exercise_id": exercise_id, "name": name},
            )
            for exercise_id, name, vector in zip(ids, names, vectors, strict=True)
        ],
        wait=True,
    )

    # Drop points for exercises that are no longer in Postgres.
    await qdrant.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must_not=[models.HasIdCondition(has_id=ids)])
        ),
        wait=True,
    )

    result = await qdrant.count(collection_name=collection, exact=True)
    return result.count


async def _ensure_collection(qdrant: AsyncQdrantClient, collection: str, size: int) -> None:
    """Create the collection for vectors of `size`; recreate it if the size changed."""
    if await qdrant.collection_exists(collection):
        info = await qdrant.get_collection(collection)
        params = info.config.params.vectors
        if isinstance(params, models.VectorParams) and params.size == size:
            return
        # The embedding model changed. The index is derived data, so rebuild it.
        await qdrant.delete_collection(collection)

    await qdrant.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )
