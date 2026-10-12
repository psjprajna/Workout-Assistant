"""Integration tests: index_exercises() against real Postgres and Qdrant.

A fake embedder stands in for OpenRouter, so no API key is needed. Its vector size is
deliberately not 384 or 1536: the collection must take its size from the vectors.
The tests use their own collection and never touch the real "exercises" one.
"""

import pytest
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.index import index_exercises
from workout_agent.settings import Settings

TEST_COLLECTION = "exercises_test"


def fake_embedder(size: int):
    async def embed(texts: list[str]) -> list[list[float]]:
        return [[float(len(text) + i) for i in range(size)] for text in texts]

    return embed


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(url=Settings().qdrant_url)
    await client.delete_collection(TEST_COLLECTION)
    try:
        yield client
    finally:
        await client.delete_collection(TEST_COLLECTION)
        await client.close()


async def _points(qdrant: AsyncQdrantClient) -> dict[int, str]:
    """exercise_id -> name, over every point (several points per exercise collapse)."""
    points, _ = await qdrant.scroll(TEST_COLLECTION, limit=100)
    return {p.payload["exercise_id"]: p.payload["name"] for p in points if p.payload}


async def _texts(qdrant: AsyncQdrantClient) -> set[tuple[int, str, str]]:
    points, _ = await qdrant.scroll(TEST_COLLECTION, limit=100)
    return {(p.payload["exercise_id"], p.payload["kind"], p.payload["text"])
            for p in points if p.payload}


async def _vector_size(qdrant: AsyncQdrantClient) -> int:
    info = await qdrant.get_collection(TEST_COLLECTION)
    return info.config.params.vectors.size  # type: ignore[union-attr]


async def test_one_point_per_exercise_with_size_from_embedder(conn, seeded, qdrant):
    count = await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)

    assert (count.exercises, count.points) == (2, 2)
    assert await _points(qdrant) == {
        2: "Barbell Bench Press",
        8: "Barbell Incline Bench Press",
    }
    assert await _vector_size(qdrant) == 5


async def test_running_twice_makes_no_duplicates(conn, seeded, qdrant):
    await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)
    count = await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)

    assert (count.exercises, count.points) == (2, 2)
    assert set(await _points(qdrant)) == {2, 8}


async def test_removed_exercise_loses_its_point(conn, seeded, qdrant):
    await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)
    await conn.execute("delete from exercise_logs where exercise_id = 8")
    await conn.execute("delete from exercises where exercise_id = 8")

    count = await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)

    assert (count.exercises, count.points) == (1, 1)
    assert set(await _points(qdrant)) == {2}


async def test_new_vector_size_recreates_collection(conn, seeded, qdrant):
    await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)
    count = await index_exercises(conn, qdrant, fake_embedder(7), TEST_COLLECTION)

    assert count.exercises == 2
    assert await _vector_size(qdrant) == 7


async def test_empty_table_indexes_nothing(conn, qdrant):
    count = await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)

    assert (count.exercises, count.points) == (0, 0)
    assert not await qdrant.collection_exists(TEST_COLLECTION)


async def test_embedder_returning_wrong_count_fails(conn, seeded, qdrant):
    async def short(texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0]]

    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        await index_exercises(conn, qdrant, short, TEST_COLLECTION)


async def test_other_names_and_description_get_their_own_points(conn, seeded, qdrant):
    await conn.execute(
        "insert into exercise_aliases (exercise_id, alias, source) values (2, 'flat bench', 'catalog')"
    )
    await conn.executemany(
        "insert into exercise_muscles (exercise_id, muscle, position) values ($1, $2, $3)",
        [(2, "chest", 0), (2, "triceps", 1)],
    )
    await conn.execute("insert into exercise_details values (2, 'barbell')")

    count = await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)

    assert (count.exercises, count.points) == (2, 4)
    assert await _texts(qdrant) == {
        (2, "name", "Barbell Bench Press"),
        (2, "alias", "flat bench"),
        (2, "description", "Barbell Bench Press: works the chest, triceps; equipment: barbell."),
        (8, "name", "Barbell Incline Bench Press"),
    }


async def test_removed_alias_loses_its_point(conn, seeded, qdrant):
    await conn.execute(
        "insert into exercise_aliases (exercise_id, alias, source) values (2, 'flat bench', 'catalog')"
    )
    await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)
    await conn.execute("delete from exercise_aliases")

    count = await index_exercises(conn, qdrant, fake_embedder(5), TEST_COLLECTION)

    assert count.points == 2
    assert (2, "alias", "flat bench") not in await _texts(qdrant)
