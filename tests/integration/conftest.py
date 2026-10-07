from datetime import date
from decimal import Decimal

import asyncpg
import pytest

from workout_agent.db.migrate import apply_migrations
from workout_agent.settings import Settings

TEST_DB = "workout_test"


def _with_database(url: str, name: str) -> str:
    return url.rsplit("/", 1)[0] + "/" + name


@pytest.fixture
def test_database_url() -> str:
    return _with_database(Settings().database_url, TEST_DB)


@pytest.fixture
async def conn():
    """A connection to the test database, migrated and empty."""
    base_url = Settings().database_url

    admin = await asyncpg.connect(base_url)
    try:
        exists = await admin.fetchval(
            "select 1 from pg_database where datname = $1", TEST_DB
        )
        if not exists:
            await admin.execute(f"create database {TEST_DB}")
    finally:
        await admin.close()

    connection = await asyncpg.connect(_with_database(base_url, TEST_DB))
    try:
        await apply_migrations(connection)
        await connection.execute("truncate sets, exercise_logs, exercises cascade")
        yield connection
    finally:
        await connection.close()


@pytest.fixture
async def seeded(conn):
    """The test database with two bench presses and one set each."""
    await conn.executemany(
        "insert into exercises (exercise_id, name) values ($1, $2)",
        [(2, "Barbell Bench Press"), (8, "Barbell Incline Bench Press")],
    )
    await conn.executemany(
        "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
        "values ($1, $2, $3)",
        [(10, 2, date(2026, 2, 8)), (11, 8, date(2026, 2, 9))],
    )
    await conn.executemany(
        "insert into sets (exercise_log_id, set_number, weight_kg, weight_raw, reps) "
        "values ($1, $2, $3, $4, $5)",
        [
            (10, 1, Decimal("80.00"), Decimal("80.00"), 8),
            (11, 1, Decimal("60.00"), Decimal("60.00"), 10),
        ],
    )