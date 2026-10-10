"""Unit tests: embed() against a mocked OpenRouter, no network and no API key needed."""

import json

import httpx
import pytest

from workout_agent.embeddings.openrouter import EMBEDDINGS_URL, MODEL_NAME, embed
from workout_agent.settings import Settings


def _settings():
    return Settings(openrouter_api_key="sk-test", _env_file=None)


def _mock_response(vectors: list[list[float]]) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vector, "index": i}
            for i, vector in enumerate(vectors)
        ],
        "model": MODEL_NAME,
    }


async def test_returns_one_vector_per_text_in_order(httpx_mock):
    httpx_mock.add_response(
        url=EMBEDDINGS_URL,
        json=_mock_response([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    )

    result = await embed(_settings(), ["bench press", "squat", "deadlift"])

    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


async def test_result_order_follows_index_not_response_order(httpx_mock):
    # The response lists index 1 before index 0; embed() must still return input order.
    httpx_mock.add_response(
        url=EMBEDDINGS_URL,
        json={
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [9.0], "index": 1},
                {"object": "embedding", "embedding": [1.0], "index": 0},
            ],
            "model": MODEL_NAME,
        },
    )

    result = await embed(_settings(), ["first", "second"])

    assert result == [[1.0], [9.0]]


async def test_sends_model_and_input(httpx_mock):
    httpx_mock.add_response(url=EMBEDDINGS_URL, json=_mock_response([[0.1]]))

    await embed(_settings(), ["bench press"])

    sent = httpx_mock.get_requests()[0]
    body = json.loads(sent.content)
    assert body["model"] == MODEL_NAME
    assert body["input"] == ["bench press"]
    assert sent.headers["Authorization"] == "Bearer sk-test"


async def test_non_200_response_raises(httpx_mock):
    httpx_mock.add_response(url=EMBEDDINGS_URL, status_code=401, json={"error": "bad key"})

    with pytest.raises(httpx.HTTPStatusError):
        await embed(_settings(), ["bench press"])


async def test_missing_api_key_raises_without_a_request(httpx_mock):
    settings = Settings(_env_file=None)
    assert settings.openrouter_api_key is None

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await embed(settings, ["bench press"])

    assert httpx_mock.get_requests() == []
