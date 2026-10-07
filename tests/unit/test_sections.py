import csv
import io
from pathlib import Path

from workout_agent.importing.sections import split_sections

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mini_jefit.csv"

#
def load():
    return split_sections(FIXTURE.read_text())

#
def test_finds_every_section_in_order():
    assert list(load()) == ["SETTING", "EXERCISE LOGS", "EXERCISE SET LOGS", "NOTES"]

#
def test_section_body_starts_at_the_header_row():
    assert load()["EXERCISE LOGS"].splitlines()[0].startswith("USERID,")

#
def test_markers_and_closing_rules_are_not_in_the_body():
    for body in load().values():
        assert "####" not in body

#
def test_quoted_commas_survive_csv_parsing():
    rows = list(csv.DictReader(io.StringIO(load()["EXERCISE LOGS"])))
    assert rows[0]["ename"] == "Bench Press, Flat"

#
def test_quoted_newlines_survive_csv_parsing():
    rows = list(csv.DictReader(io.StringIO(load()["NOTES"])))
    assert rows[0]["note"] == "Felt strong,\nslept well"

#
def test_last_section_kept_even_without_closing_rule():
    text = "### A ####\n\nx,y\n1,2\n"
    assert split_sections(text) == {"A": "x,y\n1,2"}