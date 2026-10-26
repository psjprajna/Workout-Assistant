"""How recent the logged history is: the date of the newest logged session.

Plain code, read-only. Called by code once a day (sessions/daily_checks.py), never by the
model. If nothing is logged at all, returns status "no_rows".
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

import asyncpg


@dataclass
class LatestRecordResult:
    status: Literal["ok", "no_rows"]
    latest_on: date | None


async def latest_record(conn: asyncpg.Connection) -> LatestRecordResult:
    """The date of the newest exercise log, or no_rows when nothing is logged."""
    latest_on = await conn.fetchval("SELECT max(performed_on) FROM exercise_logs")
    if latest_on is None:
        return LatestRecordResult(status="no_rows", latest_on=None)
    return LatestRecordResult(status="ok", latest_on=latest_on)
