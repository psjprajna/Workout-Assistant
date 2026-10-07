"""Look up the most recent logged session for one exercise.

Returns the exercise name, the date, and the sets performed that day.
If the exercise has no logs at all, returns status "no_rows".
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal

import asyncpg


@dataclass
class SetItem:
    set_number: int
    weight_kg: Decimal
    reps: int


@dataclass
class LastSessionResult:
    status: Literal["ok", "no_rows"]
    exercise_id: int
    exercise_name: str | None
    performed_on: date | None
    exercise_log_id: int | None
    sets: list[SetItem] = field(default_factory=list)

async def get_last_session(conn: asyncpg.Connection, exercise_id: int) -> LastSessionResult:


    """Get the last session for a specific exercise.

    Returns a LastSessionResult with status "ok" if found, or "no_rows" if not.
    """
    # Step 1: Find the latest log for the exercise
    sql_latest_log = """
        SELECT e.exercise_id, e.name, l.exercise_log_id, l.performed_on
        FROM exercises e
        JOIN exercise_logs l ON e.exercise_id = l.exercise_id
        WHERE l.exercise_id = $1
        ORDER BY l.performed_on DESC, l.exercise_log_id DESC
        LIMIT 1;
    """
    row = await conn.fetchrow(sql_latest_log, exercise_id)

    if row is None:
        return LastSessionResult(
            status="no_rows",
            exercise_id=exercise_id,
            exercise_name=None,
            performed_on=None,
            exercise_log_id=None,
            sets=[],
        )

    # Step 2: Fetch the sets for the latest log
    sql_sets = """
        SELECT set_number, weight_kg, reps
        FROM sets
        WHERE exercise_log_id = $1
        ORDER BY set_number;
    """
    sets_rows = await conn.fetch(sql_sets, row["exercise_log_id"])

    sets = [
        SetItem(
            set_number=s["set_number"],
            weight_kg=s["weight_kg"],
            reps=s["reps"],
        )
        for s in sets_rows
    ]

    return LastSessionResult(
        status="ok",
        exercise_id=row["exercise_id"],
        exercise_name=row["name"],
        performed_on=row["performed_on"],
        exercise_log_id=row["exercise_log_id"],
        sets=sets,
    )