"""Checks that run once a day, with the result stored outside any chat.

The first message of the day runs the lookup and stores its result under today's date;
every chat after that reads the stored row and does not query the history again.
"""

import json
from datetime import date
from typing import Any

import asyncpg

from workout_agent.agents.workout_agent import jsonable
from workout_agent.tools.latest_record import latest_record

LATEST_RECORD = "latest_record"


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
