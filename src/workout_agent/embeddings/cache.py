"""Cache embedding vectors in Postgres (table embedding_cache, migrations/0002).

cached() wraps an Embedder: texts seen before come from the table, only new ones go to
the model. Derived data, safe to truncate: the next run refetches what it needs.
"""

import hashlib

import asyncpg

from workout_agent.embeddings.index import Embedder


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cached(conn: asyncpg.Connection, model: str, embedder: Embedder) -> Embedder:
    """Return an Embedder that reads and fills embedding_cache for `model`."""

    async def embed(texts: list[str]) -> list[list[float]]:
        keys = [_key(text) for text in texts]
        rows = await conn.fetch(
            "select text_sha256, vector from embedding_cache "
            "where model = $1 and text_sha256 = any($2)",
            model,
            keys,
        )
        found = {row["text_sha256"]: list(row["vector"]) for row in rows}

        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in found))
        if missing:
            vectors = await embedder(missing)
            if len(vectors) != len(missing):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for {len(missing)} texts"
                )
            await conn.executemany(
                "insert into embedding_cache (model, text_sha256, vector) values ($1, $2, $3) "
                "on conflict do nothing",
                [(model, _key(t), v) for t, v in zip(missing, vectors, strict=True)],
            )
            found.update({_key(t): v for t, v in zip(missing, vectors, strict=True)})

        return [found[k] for k in keys]

    return embed
