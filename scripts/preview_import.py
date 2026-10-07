"""Preview what the parsers read from a Jefit export.

Usage: uv run python scripts/preview_import.py data/<your-file>.csv
"""

import sys
from pathlib import Path

from workout_agent.importing.parsers import parse_exercise_logs, parse_set_logs
from workout_agent.importing.sections import split_sections


def main(path: str) -> None:
    sections = split_sections(Path(path).read_text())
    print("sections:", ", ".join(sections))

    logs = parse_exercise_logs(sections["EXERCISE LOGS"])
    print(f"\nexercise logs: {len(logs.rows)} parsed, {len(logs.errors)} errors")
    for row in logs.rows[:10]:
        print(" ", row.date, row.exercise_id, row.exercise_name)

    sets = parse_set_logs(sections["EXERCISE SET LOGS"])
    print(f"\nset logs: {len(sets.rows)} parsed, {len(sets.errors)} errors")
    weighted = [row for row in sets.rows if row.weight_lbs]
    for row in weighted[:10]:
        print(
            " ",
            f"log {row.exercise_log_id}",
            f"set {row.set_index}",
            row.weight_lbs,
            "x",
            row.reps,
        )
    for error in (logs.errors + sets.errors)[:10]:
        print("ERROR", error)


if __name__ == "__main__":
    main(sys.argv[1])
