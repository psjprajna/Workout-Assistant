"""Shared fixtures for integration tests (they need Postgres running).

conn   - a connection to the separate workout_test database, migrated and emptied.
seeded - conn plus two bench presses with one set each.
This keeps tests away from your real workout data.
"""

from datetime import date
from decimal import Decimal

import asyncpg
import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from workout_agent.db.migrate import apply_migrations
from workout_agent.settings import Settings

TEST_DB = "workout_test"

# OpenRouter answers that mean "your account can't run this", not "the code is wrong".
_OPENROUTER_SKIPS = {
    402: "OpenRouter account is out of credits (HTTP 402)",
    404: "no OpenRouter provider matches the model and data policy (HTTP 404)",
}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Skip, not fail, an `llm` test when OpenRouter refuses for account reasons."""
    try:
        return (yield)
    except (ModelHTTPError, httpx.HTTPStatusError) as error:
        status = (
            error.status_code
            if isinstance(error, ModelHTTPError)
            else error.response.status_code
        )
        if item.get_closest_marker("llm") and status in _OPENROUTER_SKIPS:
            pytest.skip(_OPENROUTER_SKIPS[status])
        raise


def _with_database(url: str, name: str) -> str:
    return url.rsplit("/", 1)[0] + "/" + name


@pytest.fixture
def test_database_url() -> str:
    return _with_database(Settings().database_url, TEST_DB)


# Fixtures are set-up helpers: a test that lists one as an argument receives its value.
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