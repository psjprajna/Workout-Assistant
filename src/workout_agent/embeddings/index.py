"""Index exercises into Qdrant, for find_exercise's search by meaning (docs/tools.md).

Several points per exercise, one per text: its name, each other name (exercise_aliases),
and a description from its muscles and equipment (the catalog). A short text embeds
sharply, so "rdl" meets the point "rdl" instead of a long sentence that mentions it.
search_exercises groups the points by exercise_id, scoring each exercise by its best text.

Point ids are derived from (exercise_id, kind, text), so running the indexer again
overwrites points instead of adding duplicates, and points for texts that no longer exist
are deleted. The vector size comes from the embedding model's output, never a constant.
"""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import asyncpg
from qdrant_client import AsyncQdrantClient, models

COLLECTION = "exercises"

# How many texts go to the embedding model in one request.
BATCH_SIZE = 256

# Namespace for point ids: uuid5(namespace, "exercise_id:kind:text").
POINT_NAMESPACE = uuid.UUID("5d1f6a3e-6c55-4c1e-9f7e-0b8f6f1d2a10")

# Takes texts, returns one vector per text in the same order. The script passes the
# OpenRouter embed(); tests pass a fake so they need no API key.
Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass(frozen=True)
class IndexText:
    exercise_id: int
    name: str
    kind: str  # "name", "alias" or "description"
    text: str

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(POINT_NAMESPACE, f"{self.exercise_id}:{self.kind}:{self.text}"))


@dataclass(frozen=True)
class IndexCount:
    exercises: int  # exercises with at least one point
    points: int


def describe(name: str, muscles: list[str], equipment: str | None) -> str | None:
    """The description text, or None if the catalog says nothing about the exercise."""
    if not muscles:
        return None
    text = f"{name}: works the {', '.join(muscles)}"
    return f"{text}; equipment: {equipment}." if equipment else f"{text}."


async def index_texts(conn: asyncpg.Connection) -> list[IndexText]:
    """Every text to embed, from the exercises table and the catalog tables."""
    rows = await conn.fetch(
        """
        select e.exercise_id, e.name, d.equipment,
               coalesce((select array_agg(a.alias order by a.alias)
                         from exercise_aliases a where a.exercise_id = e.exercise_id),
                        '{}') as aliases,
               coalesce((select array_agg(m.muscle order by m.position)
                         from exercise_muscles m where m.exercise_id = e.exercise_id),
                        '{}') as muscles
        from exercises e left join exercise_details d using (exercise_id)
        order by e.exercise_id
        """
    )
    texts = []
    for r in rows:
        exercise_id, name = r["exercise_id"], r["name"]
        texts.append(IndexText(exercise_id, name, "name", name))
        texts += [IndexText(exercise_id, name, "alias", a) for a in r["aliases"]]
        description = describe(name, list(r["muscles"]), r["equipment"])
        if description:
            texts.append(IndexText(exercise_id, name, "description", description))
    return texts


async def index_exercises(
    conn: asyncpg.Connection,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    collection: str = COLLECTION,
) -> IndexCount:
    """Embed every exercise's texts into `collection`; return what the collection holds."""
    texts = await index_texts(conn)
    if not texts:
        return IndexCount(0, 0)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [t.text for t in texts[start : start + BATCH_SIZE]]
        batch_vectors = await embedder(batch)
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(batch_vectors)} vectors for {len(batch)} texts"
            )
        vectors.extend(batch_vectors)

    await _ensure_collection(qdrant, collection, len(vectors[0]))

    ids: list[models.ExtendedPointId] = [t.point_id for t in texts]
    await qdrant.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=t.point_id,
                vector=vector,
                payload={
                    "exercise_id": t.exercise_id,
                    "name": t.name,
                    "kind": t.kind,
                    "text": t.text,
                },
            )
            for t, vector in zip(texts, vectors, strict=True)
        ],
        wait=True,
    )

    # Drop points for texts that are gone: removed exercises, aliases or descriptions.
    await qdrant.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must_not=[models.HasIdCondition(has_id=ids)])
        ),
        wait=True,
    )

    points = (await qdrant.count(collection_name=collection, exact=True)).count
    return IndexCount(exercises=len({t.exercise_id for t in texts}), points=points)


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
    # search_exercises groups points by exercise_id.
    await qdrant.create_payload_index(
        collection_name=collection,
        field_name="exercise_id",
        field_schema=models.PayloadSchemaType.INTEGER,
        wait=True,
    )
