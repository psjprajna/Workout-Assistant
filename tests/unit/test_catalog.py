"""Unit tests: the exercise catalog parser and checks (no database)."""

import pytest

from workout_agent.catalog import CatalogEntry, check_catalog, load_catalog, parse_catalog


def _toml(**entry):
    base = {"id": 199, "aliases": ["RDL"], "muscles": ["hamstrings"], "equipment": "barbell"}
    return {"exercise": [base | entry]}


def test_aliases_are_tidied_like_a_query():
    catalog = parse_catalog(_toml(aliases=["RDL", "DB Romanian-Deadlift"]))
    assert catalog[199].aliases == ("rdl", "dumbbell romanian deadlift")


@pytest.mark.parametrize(
    "data",
    [
        {"exercises": []},
        {"exercise": []},
        _toml(id="199"),
        _toml(extra=1),
        _toml(aliases=["rdl", "RDL"]),
        _toml(muscles=[]),
        _toml(muscles=["hammies"]),
        _toml(equipment=" "),
        {"exercise": _toml()["exercise"] * 2},
    ],
    ids=[
        "unknown top key",
        "no entries",
        "id not int",
        "unknown key",
        "alias repeated once tidied",
        "no muscles",
        "unknown muscle",
        "blank equipment",
        "duplicate id",
    ],
)
def test_parse_catalog_rejects_bad_input(data):
    with pytest.raises(ValueError):
        parse_catalog(data)


def test_check_catalog_finds_missing_ids_and_alias_clashes():
    catalog = {
        44: CatalogEntry(44, ("lying leg raise",), ("abs",), "bodyweight"),
        61: CatalogEntry(61, ("leg raise",), ("abs",), "bodyweight"),
        7: CatalogEntry(7, (), ("abs",), "bodyweight"),
    }
    names = {44: "Leg Raise", 61: "Hanging Leg Raise", 631: "Plank"}
    assert check_catalog(catalog, names) == [
        "id 7 is not in the exercises table",
        "exercise 631 (Plank) has no catalog entry",
        "id 61: alias 'leg raise' is the name of [44]",
    ]


def test_checked_in_catalog_parses():
    assert len(load_catalog()) >= 150
