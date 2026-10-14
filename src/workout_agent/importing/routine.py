"""Read the weekly plan from the ROUTINES section and store it in Postgres.

The section holds three kinds of table, each row under its own repeated header: the
routine ("PPL"), its days ("Push A", ...) and the exercises planned on each day with a
target rep count per set. The header's columns say which kind a row is. A bad row is
reported in Routine.errors and skipped; it never stops the import.

load_routine replaces the stored plan in one transaction: the plan changes when it is
edited in Jefit, so a new export's plan is the whole truth. Exercises planned but never
logged are added to the exercises table; names already there are kept.
"""

import csv
import io
from dataclasses import dataclass, field

import asyncpg
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .parsers import _describe  # pyright: ignore[reportPrivateUsage]


def _clean(name: str) -> str:
    """Remove leading, trailing and doubled spaces."""
    return " ".join(name.split())


class RoutineInfo(BaseModel):
    id: int = Field(validation_alias="_id")
    name: str

    model_config = {"validate_by_name": True}


class PlanDay(BaseModel):
    """One day of a routine. Days have a position, not a date."""

    id: int = Field(validation_alias="_id")
    routine_id: int = Field(validation_alias="package")
    name: str
    position: int = Field(validation_alias="dayIndex")
    in_rotation: bool = Field(validation_alias="rest_day")

    model_config = {"validate_by_name": True}

    @field_validator("name", mode="after")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return _clean(value)

    @field_validator("in_rotation", mode="before")
    @classmethod
    def not_a_rest_day(cls, value: object) -> object:
        # The column is rest_day; a day is in the rotation when it is not a rest day.
        if isinstance(value, str) and value.strip() in {"0", "1"}:
            return value.strip() == "0"
        return value


class PlannedExercise(BaseModel):
    """One exercise on one day, with a target rep count for each set."""

    id: int = Field(validation_alias="_id")
    day_id: int = Field(validation_alias="belongplan")
    exercise_id: int
    exercise_name: str = Field(validation_alias="exercisename")
    position: int = Field(validation_alias="mysort")
    target_reps: list[int] = Field(validation_alias="targetrep", min_length=1)
    set_count: int | None = Field(default=None, validation_alias="setcount", exclude=True)

    model_config = {"validate_by_name": True}

    @field_validator("exercise_name", mode="after")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return _clean(value)

    @field_validator("target_reps", mode="before")
    @classmethod
    def split_targets(cls, value: object) -> object:
        if isinstance(value, str):  # "7,8,10"
            return [part.strip() for part in value.split(",")]
        return value

    @model_validator(mode="after")
    def one_target_per_set(self) -> "PlannedExercise":
        if self.set_count is not None and self.set_count != len(self.target_reps):
            raise ValueError(
                f"{self.set_count} sets but {len(self.target_reps)} target reps"
            )
        return self


@dataclass
class Routine:
    routines: list[RoutineInfo] = field(default_factory=list)
    days: list[PlanDay] = field(default_factory=list)
    planned: list[PlannedExercise] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _kind(header: list[str]) -> type[BaseModel] | None:
    if "belongplan" in header:
        return PlannedExercise
    if "package" in header:
        return PlanDay
    if "dayaweek" in header:
        return RoutineInfo
    return None


def parse_routines(section_text: str) -> Routine:
    """Days and planned exercises from the ROUTINES section text."""
    result = Routine()
    header: list[str] = []
    # csv.reader, not line splitting: a quoted description can span lines.
    for row_number, row in enumerate(csv.reader(io.StringIO(section_text)), start=1):
        if not row or not any(cell.strip() for cell in row):
            continue
        if row[0] == "row_id":
            header = row
            continue
        model = _kind(header)
        if model is None:
            result.errors.append(f"row {row_number}: no known header above it")
            continue
        try:
            item = model.model_validate(dict(zip(header, row, strict=False)))
        except ValidationError as exc:
            result.errors.append(_describe(row_number, exc))
            continue
        if isinstance(item, PlannedExercise):
            result.planned.append(item)
        elif isinstance(item, PlanDay):
            result.days.append(item)
        elif isinstance(item, RoutineInfo):
            result.routines.append(item)

    # A planned exercise must belong to a day in this export.
    day_ids = {d.id for d in result.days}
    for p in [p for p in result.planned if p.day_id not in day_ids]:
        result.errors.append(f"planned exercise {p.id}: day {p.day_id} is not in the export")
        result.planned.remove(p)
    return result


async def load_routine(conn: asyncpg.Connection, routine: Routine) -> None:
    """Replace the stored plan with this one, in one transaction."""
    async with conn.transaction():
        await conn.executemany(
            "insert into exercises (exercise_id, name) values ($1, $2) "
            "on conflict (exercise_id) do nothing",
            sorted({(p.exercise_id, p.exercise_name) for p in routine.planned}),
        )
        await conn.execute("delete from routines")  # cascades to days and planned
        await conn.executemany(
            "insert into routines (routine_id, name) values ($1, $2)",
            [(r.id, r.name) for r in routine.routines],
        )
        await conn.executemany(
            "insert into routine_days (day_id, routine_id, name, position, in_rotation) "
            "values ($1, $2, $3, $4, $5)",
            [(d.id, d.routine_id, d.name, d.position, d.in_rotation) for d in routine.days],
        )
        await conn.executemany(
            "insert into planned_exercises "
            "(planned_id, day_id, exercise_id, position, target_reps) "
            "values ($1, $2, $3, $4, $5)",
            [(p.id, p.day_id, p.exercise_id, p.position, p.target_reps) for p in routine.planned],
        )
