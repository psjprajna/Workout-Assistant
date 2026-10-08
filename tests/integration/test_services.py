"""Integration tests: Postgres and Qdrant from docker compose are reachable."""

import asyncpg
import httpx


async def test_postgres_is_reachable():
    conn = await asyncpg.connect("postgresql://workout:workout@localhost:5433/workout")
    try:
        assert await conn.fetchval("select 1") == 1
    finally:
        await conn.close()


async def test_qdrant_is_reachable():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:6333/healthz")
    assert response.status_code == 200