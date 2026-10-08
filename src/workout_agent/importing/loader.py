"""
Turn parsed Jefit rows into a LoadPlan and insert it into Postgres.
"""

from dataclasses import dataclass, field
from datetime import date

from .parsers import ExerciseLog
from .sets import parse_logged_sets, weight_in_kg


@dataclass
class LogRow:
    exercise_log_id: int
    exercise_id: int
    performed_on: date


@dataclass
class SetRow:
    exercise_log_id: int
    set_number: int
    weight_kg: float
    weight_raw: float
    reps: int


@dataclass
class LoadPlan:
    exercises: dict[int, str] = field(default_factory=dict)
    logs: list[LogRow] = field(default_factory=list)
    sets: list[SetRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

def _clean_name(name: str) -> str:
    """Remove leading, trailing and doubled spaces."""
    return " ".join(name.split())

def build_plan(
    logs: list[ExerciseLog],
    timed_exercise_ids: frozenset[int] = frozenset(),
) -> LoadPlan:

    
    """
    Builds a LoadPlan from a list of log input objects.
    Processes exercise logs, parses their raw sets data, 
    and handles data conversions or errors gracefully.
    """
    plan = LoadPlan()

    for log in logs:
        # Create and add the LogRow first (the log still exists even if sets are bad)
        log_row = LogRow(
            exercise_log_id=log.id,
            exercise_id=log.exercise_id,
            performed_on=log.date,
        )
        plan.logs.append(log_row)
        plan.exercises.setdefault(log.exercise_id, _clean_name(log.exercise_name))

        if log.exercise_id in timed_exercise_ids:
            continue  # duration logs: keep the log, skip the sets

        try:
            # Parse raw sets text into a list of (weight, reps) tuples
            parsed_sets = parse_logged_sets(log.logs)
            
            # Enumerate to assign sequential set numbers starting at 1
            for set_number, (weight_raw, reps) in enumerate(parsed_sets, start=1):
                # Convert the raw weight to KG based on the performance date rules
                weight_kg = weight_in_kg(weight_raw, log.date)
                
                set_row = SetRow(
                    exercise_log_id=log.id,
                    set_number=set_number,
                    weight_kg=weight_kg,
                    weight_raw=weight_raw,
                    reps=reps
                )
                plan.sets.append(set_row)
                
        except ValueError as exc:
            # Catch parsing errors, record them, and move to the next log
            plan.errors.append(f"log {log.id}: {exc}")

    return plan

async def load_plan(conn, plan: LoadPlan) -> None:


    """Insert a LoadPlan in one transaction. Safe to run twice."""
    exercise_rows = list(plan.exercises.items())

    log_rows = [
        (log.exercise_log_id, log.exercise_id, log.performed_on)
        for log in plan.logs
    ]

    set_rows = [
        (s.exercise_log_id, s.set_number, s.weight_kg, s.weight_raw, s.reps)
        for s in plan.sets
    ]

    async with conn.transaction():
        await conn.executemany(
            """
            INSERT INTO exercises (exercise_id, name)
            VALUES ($1, $2)
            ON CONFLICT (exercise_id) DO NOTHING
            """,
            exercise_rows,
        )
        await conn.executemany(
            """
            INSERT INTO exercise_logs (exercise_log_id, exercise_id, performed_on)
            VALUES ($1, $2, $3)
            ON CONFLICT (exercise_log_id) DO NOTHING
            """,
            log_rows,
        )
        await conn.executemany(
            """
            INSERT INTO sets
                (exercise_log_id, set_number, weight_kg, weight_raw, reps)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (exercise_log_id, set_number) DO NOTHING
            """,
            set_rows,
        )