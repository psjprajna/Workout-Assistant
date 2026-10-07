import asyncpg

from workout_agent.db.migrate import apply_migrations
from workout_agent.settings import Settings


async def _connect():
    return await asyncpg.connect(Settings().database_url)


async def test_tables_exist_after_migration():
    conn = await _connect()
    try:
        await apply_migrations(conn)
        rows = await conn.fetch(
            "select table_name from information_schema.tables "
            "where table_schema = 'public'"
        )
        names = {r["table_name"] for r in rows}
        assert {"exercises", "exercise_logs", "sets"} <= names
    finally:
        await conn.close()


async def test_apply_migrations_twice_is_safe():
    conn = await _connect()
    try:
        await apply_migrations(conn)
        await apply_migrations(conn)  # must not raise
    finally:
        await conn.close()