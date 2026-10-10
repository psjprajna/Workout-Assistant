"""Integration tests: search_exercises() against real Qdrant.

A fake embedder stands in for OpenRouter: each name maps to a fixed 3-d vector, so the
scores are known. The tests use their own collection and never touch "exercises".
"""

import pytest
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.index import index_exercises
from workout_agent.embeddings.search import search_exercises
from workout_agent.settings import Settings

TEST_COLLECTION = "exercises_search_test"

VECTORS = {
    "Barbell Bench Press": [1.0, 0.0, 0.0],
    "Barbell Incline Bench Press": [0.8, 0.6, 0.0],
    "Barbell Curl": [0.0, 0.0, 1.0],
    "chest press": [1.0, 0.1, 0.0],
}


async def fake_embedder(texts: list[str]) -> list[list[float]]:
    return [VECTORS[text] for text in texts]


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(url=Settings().qdrant_url)
    await client.delete_collection(TEST_COLLECTION)
    try:
        yield client
    finally:
        await client.delete_collection(TEST_COLLECTION)
        await client.close()


@pytest.fixture
async def indexed(conn, qdrant):
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [(2, "Barbell Bench Press"), (8, "Barbell Incline Bench Press"), (112, "Barbell Curl")],
    )
    await index_exercises(conn, qdrant, fake_embedder, TEST_COLLECTION)


async def test_results_are_best_first_with_ids_names_and_scores(indexed, qdrant):
    hits = await search_exercises(qdrant, fake_embedder, "chest press", 3, TEST_COLLECTION)

    assert [(h.exercise_id, h.name) for h in hits] == [
        (2, "Barbell Bench Press"),
        (8, "Barbell Incline Bench Press"),
        (112, "Barbell Curl"),
    ]
    assert hits[0].score > hits[1].score > hits[2].score
    assert hits[0].score == pytest.approx(0.995, abs=0.001)


async def test_limit_is_respected(indexed, qdrant):
    hits = await search_exercises(qdrant, fake_embedder, "chest press", 1, TEST_COLLECTION)

    assert [h.exercise_id for h in hits] == [2]


async def test_missing_collection_returns_nothing(qdrant):
    assert await search_exercises(qdrant, fake_embedder, "chest press", 3, TEST_COLLECTION) == []


async def test_blank_query_returns_nothing_without_embedding(indexed, qdrant):
    async def must_not_embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("the embedder was called for a blank query")

    assert await search_exercises(qdrant, must_not_embed, "   ", 3, TEST_COLLECTION) == []
