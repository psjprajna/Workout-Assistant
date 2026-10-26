"""Integration tests: the Postgres embedding cache, with a counting fake embedder."""

from workout_agent.embeddings.cache import cached


def counting_embedder():
    calls: list[list[str]] = []

    async def embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(text)), 0.5] for text in texts]

    return embed, calls


async def test_second_call_is_served_from_the_cache(conn):
    await conn.execute("truncate embedding_cache")
    fake, calls = counting_embedder()
    embed = cached(conn, "test-model", fake)

    first = await embed(["squat", "bench press"])
    second = await embed(["bench press", "squat"])

    assert calls == [["squat", "bench press"]]
    assert second == [first[1], first[0]]


async def test_only_new_texts_reach_the_model_and_duplicates_once(conn):
    await conn.execute("truncate embedding_cache")
    fake, calls = counting_embedder()
    embed = cached(conn, "test-model", fake)

    await embed(["squat"])
    result = await embed(["squat", "row", "row"])

    assert calls == [["squat"], ["row"]]
    assert result[1] == result[2] == [3.0, 0.5]


async def test_models_do_not_share_vectors(conn):
    await conn.execute("truncate embedding_cache")
    fake, calls = counting_embedder()

    await cached(conn, "model-a", fake)(["squat"])
    await cached(conn, "model-b", fake)(["squat"])

    assert calls == [["squat"], ["squat"]]
