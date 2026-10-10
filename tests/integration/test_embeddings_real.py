"""Integration tests: embed() against a REAL OpenRouter embedding model.

These call OpenRouter, so they cost a little money and need OPENROUTER_API_KEY in .env.
They are marked `llm`: run them with `uv run pytest -m llm`, skip with `-m "not llm"`.
"""

import math

import pytest
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.index import index_exercises
from workout_agent.embeddings.openrouter import embed
from workout_agent.embeddings.search import MIN_SCORE, search_exercises
from workout_agent.settings import Settings

TEST_COLLECTION = "exercises_real_test"

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        Settings().openrouter_api_key is None, reason="OPENROUTER_API_KEY is not set"
    ),
]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


async def test_returns_one_equal_length_vector_per_text():
    vectors = await embed(Settings(), ["barbell bench press", "barbell squat"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0


async def test_similar_meanings_score_closer():
    target, close, far = await embed(
        Settings(),
        ["barbell bench press", "chest press machine", "banana"],
    )

    assert _cosine_similarity(target, close) > _cosine_similarity(target, far)


async def _real_embedder(texts: list[str]) -> list[list[float]]:
    return await embed(Settings(), texts)


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(url=Settings().qdrant_url)
    await client.delete_collection(TEST_COLLECTION)
    try:
        yield client
    finally:
        await client.delete_collection(TEST_COLLECTION)
        await client.close()


async def test_min_score_separates_real_matches_from_noise(conn, qdrant):
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [(2, "Barbell Bench Press"), (8, "Barbell Incline Bench Press"), (112, "Barbell Curl")],
    )
    await index_exercises(conn, qdrant, _real_embedder, TEST_COLLECTION)

    chest = await search_exercises(qdrant, _real_embedder, "chest press", 3, TEST_COLLECTION)
    noise = await search_exercises(qdrant, _real_embedder, "zzz", 3, TEST_COLLECTION)

    by_id = {hit.exercise_id: hit.score for hit in chest}
    assert by_id[2] >= MIN_SCORE
    assert all(hit.score < MIN_SCORE for hit in noise)
