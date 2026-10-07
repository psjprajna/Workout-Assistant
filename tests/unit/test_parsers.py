from datetime import date
from pathlib import Path

from workout_agent.importing.parsers import parse_exercise_logs, parse_set_logs
from workout_agent.importing.sections import split_sections

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mini_jefit.csv"


def sections():
    return split_sections(FIXTURE.read_text())


def test_exercise_logs_become_typed_rows():
    result = parse_exercise_logs(sections()["EXERCISE LOGS"])
    first = result.rows[0]
    assert (first.id, first.exercise_id) == (1, 10)
    assert first.exercise_name == "Bench Press, Flat"
    assert first.date == date(2026, 1, 5)
    assert result.errors == []


def test_set_logs_become_typed_rows():
    result = parse_set_logs(sections()["EXERCISE SET LOGS"])
    first = result.rows[0]
    assert (first.exercise_log_id, first.set_index) == (1, 0)
    assert first.weight_lbs == 135.0
    assert first.reps == 8


def test_blank_weight_and_reps_become_none():
    text = "_id,exercise_log_id,set_index,weight_lbs,reps\n9,1,0,,\n"
    row = parse_set_logs(text).rows[0]
    assert row.weight_lbs is None and row.reps is None


def test_bad_row_is_reported_not_raised():
    text = "_id,exercise_log_id,set_index,weight_lbs,reps\n1,1,0,135,8\n2,1,1,135,abc\n"
    result = parse_set_logs(text)
    assert len(result.rows) == 1
    assert len(result.errors) == 1
    assert "abc" in result.errors[0] or "reps" in result.errors[0]