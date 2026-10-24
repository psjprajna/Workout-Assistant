"""Checks that run once a day, with the result stored outside any chat.

The first message of the day runs the lookup and stores its result under today's date;
every chat after that reads the stored row and does not query the history again.
"""

import json
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import asyncpg

from workout_agent.agents.workout_agent import jsonable
from workout_agent.tools.latest_record import latest_record

LATEST_RECORD = "latest_record"
JEFIT_REFRESH = "jefit_refresh"
REFRESH_LOCK = 7_204_202  # advisory lock key: one chat refreshes while the others go on


async def _stored(conn: asyncpg.Connection, name: str, today: date) -> dict[str, Any] | None:
    raw = await conn.fetchval(
        "SELECT result FROM daily_checks WHERE check_name = $1 AND checked_on = $2", name, today
    )
    return None if raw is None else json.loads(raw)


async def latest_record_today(
    conn: asyncpg.Connection, today: date
) -> tuple[dict[str, Any], bool]:
    """Today's latest_record result, and whether this call looked it up (False: read it)."""
    stored = await _stored(conn, LATEST_RECORD, today)
    if stored is not None:
        return stored, False
    result = jsonable(await latest_record(conn))
    # Two first messages at once both look up; the primary key keeps one row, and both
    # return that one.
    await conn.execute(
        "INSERT INTO daily_checks (check_name, checked_on, result) VALUES ($1, $2, $3) "
        "ON CONFLICT DO NOTHING",
        LATEST_RECORD,
        today,
        json.dumps(result),
    )
    return await _stored(conn, LATEST_RECORD, today) or result, True


async def refresh_today(
    conn: asyncpg.Connection, today: date, refresh: Callable[[], Awaitable[dict[str, Any]]]
) -> tuple[dict[str, Any] | None, bool]:
    """Today's Jefit refresh result, and whether this call ran it (False: read it).

    The first message of the day runs `refresh` and stores its result, a failure too, so a
    wrong password is not retried (and waited for) on every message. While one chat is
    refreshing, another gets None and just carries on with the data it has.
    """
    stored = await _stored(conn, JEFIT_REFRESH, today)
    if stored is not None:
        return stored, False
    if not await conn.fetchval("SELECT pg_try_advisory_lock($1)", REFRESH_LOCK):
        return None, False
    try:
        stored = await _stored(conn, JEFIT_REFRESH, today)  # finished while we waited
        if stored is not None:
            return stored, False
        result = await refresh()
        await conn.execute(
            "INSERT INTO daily_checks (check_name, checked_on, result) VALUES ($1, $2, $3) "
            "ON CONFLICT DO NOTHING",
            JEFIT_REFRESH,
            today,
            json.dumps(result),
        )
        return result, True
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", REFRESH_LOCK)
