"""Which plan day to train next, and what is on a day.

The plan days have numbers, not dates, and Jefit does not record which day a recent
session was (the export's session rows say day 0). So each date's logged exercises are
compared with each day's planned exercises: the share of exercises in common (shared
over all of either), at least MIN_SHARE and strictly ahead of every other day, or the
date is left unmatched. Arms is matched too, so an arms session is not mistaken for a
pull day, but only rotation days are suggested. The suggestion is the rotation day done
longest ago (never done comes first); a tie goes to the day usually done on today's
weekday, then to plan order. Pure code, no model.

workout_days is the tool the model calls: small, so it costs few tokens. planned_day is
called by code once the model has chosen a day; its full list never goes to the model.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import asyncpg

from workout_agent.importing.timed_exercises import TIMED_EXERCISE_IDS
from workout_agent.tools.get_last_session import SetItem

MIN_SHARE = 0.25  # measured on the real log: below this a match is mostly noise


def short_day_name(name: str) -> str:
    """The day name without Jefit's note: "Push A. Chest and shoulder" -> "Push A"."""
    return name.split(".")[0].strip()


@dataclass(frozen=True)
class PlanDayInfo:
    day_id: int
    name: str
    position: int
    in_rotation: bool
    exercise_ids: frozenset[int]


def match_sessions(
    sessions: Mapping[date, set[int]], days: Sequence[PlanDayInfo]
) -> dict[date, int]:
    """date -> day_id for each session that clearly matches one plan day."""
    matched: dict[date, int] = {}
    for when, logged in sessions.items():
        scores = sorted(
            (
                (len(logged & d.exercise_ids) / len(logged | d.exercise_ids), d.day_id)
                for d in days
                if logged | d.exercise_ids
            ),
            reverse=True,
        )
        if not scores or scores[0][0] < MIN_SHARE:
            continue
        if len(scores) > 1 and scores[1][0] == scores[0][0]:
            continue  # a tie: can't tell which day it was
        matched[when] = scores[0][1]
    return matched


def suggest_day(
    days: Sequence[PlanDayInfo],
    last_done: Mapping[int, date],
    usual_weekday: Mapping[int, int],
    today: date,
) -> int | None:
    """The rotation day to train next: done longest ago, then today's weekday, then order."""
    rotation = [d for d in days if d.in_rotation]
    if not rotation:
        return None
    best = min(
        rotation,
        key=lambda d: (
            last_done.get(d.day_id, date.min),
            0 if usual_weekday.get(d.day_id) == today.weekday() else 1,
            d.position,
        ),
    )
    return best.day_id


@dataclass
class DaySummary:
    day_id: int
    name: str
    in_rotation: bool
    last_done: date | None


@dataclass
class WorkoutDaysResult:
    status: Literal["ok", "no_rows"]  # no_rows: no plan imported
    suggested_day_id: int | None
    data_through: date | None  # the newest date in the log: the data's freshness
    days: list[DaySummary] = field(default_factory=list)


async def _plan_days(conn: asyncpg.Connection) -> list[PlanDayInfo]:
    rows = await conn.fetch(
        """
        SELECT d.day_id, d.name, d.position, d.in_rotation,
               coalesce(array_agg(p.exercise_id) FILTER (WHERE p.exercise_id IS NOT NULL),
                        '{}') AS ids
        FROM routine_days d
        LEFT JOIN planned_exercises p USING (day_id)
        GROUP BY d.day_id
        ORDER BY d.position
        """
    )
    return [
        PlanDayInfo(r["day_id"], r["name"], r["position"], r["in_rotation"], frozenset(r["ids"]))
        for r in rows
    ]


async def workout_days(conn: asyncpg.Connection, today: date) -> WorkoutDaysResult:
    """Every plan day with the date it was last done, and the day to train next."""
    days = await _plan_days(conn)
    rows = await conn.fetch(
        "SELECT performed_on, array_agg(DISTINCT exercise_id) AS ids "
        "FROM exercise_logs GROUP BY performed_on"
    )
    sessions = {r["performed_on"]: set(r["ids"]) for r in rows}
    data_through = max(sessions, default=None)
    if not days:
        return WorkoutDaysResult("no_rows", None, data_through)

    matched = match_sessions(sessions, days)
    last_done: dict[int, date] = {}
    weekdays: dict[int, Counter[int]] = {}
    for when, day_id in matched.items():
        last_done[day_id] = max(when, last_done.get(day_id, date.min))
        weekdays.setdefault(day_id, Counter())[when.weekday()] += 1
    usual = {day_id: c.most_common(1)[0][0] for day_id, c in weekdays.items()}

    return WorkoutDaysResult(
        status="ok",
        suggested_day_id=suggest_day(days, last_done, usual, today),
        data_through=data_through,
        days=[
            DaySummary(d.day_id, d.name, d.in_rotation, last_done.get(d.day_id))
            for d in days
        ],
    )


@dataclass
class PlannedItem:
    exercise_id: int
    exercise_name: str
    timed: bool  # targets are durations, not reps (importing/timed_exercises.py)
    target_reps: list[int]  # one per set
    last_on: date | None  # the latest log of this exercise, on any day
    last_sets: list[SetItem] = field(default_factory=list)


@dataclass
class PlannedDayResult:
    status: Literal["ok", "no_rows"]
    day_id: int
    day_name: str | None
    exercises: list[PlannedItem] = field(default_factory=list)


async def planned_day(conn: asyncpg.Connection, day_id: int) -> PlannedDayResult:
    """The exercises planned on one day, in order, with targets and the last session."""
    name = await conn.fetchval("SELECT name FROM routine_days WHERE day_id = $1", day_id)
    if name is None:
        return PlannedDayResult("no_rows", day_id, None)
    rows = await conn.fetch(
        """
        SELECT p.exercise_id, e.name, p.target_reps, l.exercise_log_id, l.performed_on
        FROM planned_exercises p
        JOIN exercises e USING (exercise_id)
        LEFT JOIN LATERAL (
            SELECT exercise_log_id, performed_on FROM exercise_logs
            WHERE exercise_id = p.exercise_id
            ORDER BY performed_on DESC, exercise_log_id DESC
            LIMIT 1
        ) l ON true
        WHERE p.day_id = $1
        ORDER BY p.position
        """,
        day_id,
    )
    log_ids = [r["exercise_log_id"] for r in rows if r["exercise_log_id"] is not None]
    sets: dict[int, list[SetItem]] = {}
    for s in await conn.fetch(
        "SELECT exercise_log_id, set_number, weight_kg, reps FROM sets "
        "WHERE exercise_log_id = ANY($1) ORDER BY exercise_log_id, set_number",
        log_ids,
    ):
        sets.setdefault(s["exercise_log_id"], []).append(
            SetItem(s["set_number"], s["weight_kg"], s["reps"])
        )
    return PlannedDayResult(
        status="ok",
        day_id=day_id,
        day_name=name,
        exercises=[
            PlannedItem(
                exercise_id=r["exercise_id"],
                exercise_name=r["name"],
                timed=r["exercise_id"] in TIMED_EXERCISE_IDS,
                target_reps=list(r["target_reps"]),
                last_on=r["performed_on"],
                last_sets=sets.get(r["exercise_log_id"], []),
            )
            for r in rows
        ],
    )
