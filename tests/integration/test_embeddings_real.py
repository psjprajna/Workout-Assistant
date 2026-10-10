"""Integration tests: embed() against a REAL OpenRouter embedding model.

These call OpenRouter, so they cost a little money and need OPENROUTER_API_KEY in .env.
They are marked `llm`: run them with `uv run pytest -m llm`, skip with `-m "not llm"`.
"""

import math

import pytest

from workout_agent.embeddings.openrouter import embed
from workout_agent.settings import Settings

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
