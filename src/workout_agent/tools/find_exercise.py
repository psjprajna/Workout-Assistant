"""
This module provides a function to find exercises in the database based on a query string.
The search is case-insensitive and supports partial matches. It also prioritizes exact matches
over partial matches and uses the frequency of exercise usage in logs as a tie-breaker.
"""
from dataclasses import dataclass
from typing import Literal

import asyncpg


@dataclass
class ExerciseMatch:
    exercise_id: int
    name: str


@dataclass
class FindResult:
    status: Literal["ok", "ambiguous", "no_rows"]
    matches: list[ExerciseMatch]


async def find_exercise(conn: asyncpg.Connection, query: str) -> FindResult:
    """Turn what the user typed into one of: ok, ambiguous, no_rows."""

    # Step 1: tidy the input. If nothing is left, there is nothing to look for.
    clean_query = query.strip()
    if not clean_query:
        return FindResult(status="no_rows", matches=[])

    # Step 2: turn "incline bench" into ["%incline%", "%bench%"].
    # % means "anything here" in SQL, so each pattern matches a name that
    # contains that word anywhere.
    words = clean_query.split()
    like_patterns = [f"%{word}%" for word in words]

    # Step 3: ask Postgres for every exercise whose name contains ALL the words.
    #  - ILIKE ALL($1): case-insensitive, every pattern must match.
    #  - match_priority: 1 if the whole name equals what the user typed,
    #    otherwise 2. We use it to spot exact matches.
    #  - usage_count: how many times you logged it. Used to sort, so the
    #    exercises you do most often come first in the "which one?" list.
    sql = """
        SELECT
            e.exercise_id,
            e.name,
            CASE
                WHEN LOWER(e.name) = LOWER($2) THEN 1
                ELSE 2
            END AS match_priority,
            COUNT(l.exercise_id) AS usage_count
        FROM exercises e
        LEFT JOIN exercise_logs l ON e.exercise_id = l.exercise_id
        WHERE e.name ILIKE ALL($1)
        GROUP BY e.exercise_id, e.name
        ORDER BY match_priority ASC, usage_count DESC, e.name ASC;
    """
    rows = await conn.fetch(sql, like_patterns, clean_query)

    # Step 4: nothing matched at all. This guard must come BEFORE anything
    # that assumes there is at least one row.
    if not rows:
        return FindResult(status="no_rows", matches=[])

    # Step 5: turn database rows into ExerciseMatch objects.
    matches = [
        ExerciseMatch(exercise_id=row["exercise_id"], name=row["name"])
        for row in rows
    ]

    # Step 6: pick the exact-name matches out of that list.
    # zip pairs each match with its row so we can read match_priority.
    exact = [
        m for m, row in zip(matches, rows, strict=True)
        if row["match_priority"] == 1
    ]

    # Exactly one exercise has this exact name: no doubt, we're done.
    if len(exact) == 1:
        return FindResult(status="ok", matches=exact)

    # Step 7: otherwise we may have to ask the user.
    # If several exercises share the exact name (your two Ab Rollouts), ask
    # about just those. If none matched exactly, ask about all partial matches.
    candidates = exact or matches
    if len(candidates) > 1:
        return FindResult(status="ambiguous", matches=candidates)

    # Step 8: no exact match, but only one partial match: that's our answer.
    return FindResult(status="ok", matches=candidates)