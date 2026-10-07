"""Turn raw CSV section text into validated rows.

A bad row never stops an import: it is described in ParseResult.errors and the
parser carries on. Row numbers count data rows from 1 (the header is not counted).
Weights stay in the file's own unit (lbs). Conversion to kg happens later.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class ExerciseLog(BaseModel):
    """One exercise performed in one session (a row of EXERCISE LOGS)."""

    id: int = Field(alias="_id")
    date: Date = Field(alias="mydate")
    exercise_id: int = Field(alias="eid")
    exercise_name: str = Field(alias="ename")


class SetLog(BaseModel):
    """One set of one exercise log (a row of EXERCISE SET LOGS)."""

    id: int = Field(alias="_id")
    exercise_log_id: int
    set_index: int
    weight_lbs: float | None = None
    reps: int | None = None

    @field_validator("weight_lbs", "reps", mode="before")
    @classmethod
    def blank_to_none(cls, value: Any) -> Any:
        # Cardio sets have empty weight and reps cells.
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


@dataclass
class ParseResult[T]:
    rows: list[T] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

def _describe(row_number: int, exc: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']} (got {err['input']!r})"
        for err in exc.errors()
    ]
    return f"row {row_number}: " + "; ".join(problems)


def _parse[T: BaseModel](csv_text: str, model: type[T]) -> ParseResult[T]:
    result: ParseResult[T] = ParseResult()
    for row_number, row in enumerate(csv.DictReader(io.StringIO(csv_text)), start=1):
        try:
            result.rows.append(model.model_validate(row))
        except ValidationError as exc:
            result.errors.append(_describe(row_number, exc))
    return result


def parse_exercise_logs(csv_text: str) -> ParseResult[ExerciseLog]:
    return _parse(csv_text, ExerciseLog)


def parse_set_logs(csv_text: str) -> ParseResult[SetLog]:
    return _parse(csv_text, SetLog)
