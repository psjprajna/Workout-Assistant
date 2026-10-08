"""Integration tests: migrations create the tables and can run twice."""

from workout_agent.db.migrate import apply_migrations


async def test_tables_exist_after_migration(conn):
    rows = await conn.fetch(
        "select table_name from information_schema.tables "
        "where table_schema = 'public'"
    )
    names = {r["table_name"] for r in rows}
    assert {"exercises", "exercise_logs", "sets"} <= names


async def test_apply_migrations_twice_is_safe(conn):
    await apply_migrations(conn)
    await apply_migrations(conn)  # must not raise