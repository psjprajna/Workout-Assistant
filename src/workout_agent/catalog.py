"""The exercise catalog (config/exercise_catalog.toml): other names, muscles, equipment.

parse_catalog() checks the file's shape; check_catalog() checks it against the exercises
in the database; store_catalog() writes aliases, muscles and equipment. Aliases
are stored tidied the way find_exercise tidies a phrase ("bb bench" -> "barbell bench").
"""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg

from workout_agent.tools.find_exercise import normalise

CATALOG = Path(__file__).resolve().parents[2] / "config" / "exercise_catalog.toml"

# The muscle names the catalog may use, so a typo cannot hide an exercise.
MUSCLES = frozenset(
    {"chest", "upper chest", "lower chest", "shoulders", "front delts", "side delts",
     "rear delts", "biceps", "triceps", "forearms", "lats", "upper back", "traps",
     "lower back", "abs", "obliques", "hip flexors", "glutes", "quads", "hamstrings",
     "calves", "adductors", "abductors", "cardio"}
)


@dataclass(frozen=True)
class CatalogEntry:
    exercise_id: int
    aliases: tuple[str, ...]  # tidied with normalise()
    muscles: tuple[str, ...]  # main muscle first
    equipment: str


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def parse_catalog(data: Mapping[str, Any]) -> dict[int, CatalogEntry]:
    """Turn the parsed TOML into entries by id; raise ValueError on anything malformed."""
    unknown_top = set(data) - {"exercise"}
    if unknown_top:
        raise ValueError(f"unknown top-level keys: {sorted(unknown_top)}")

    entries: dict[int, CatalogEntry] = {}
    for number, raw in enumerate(data.get("exercise", []), start=1):
        unknown = set(raw) - {"id", "aliases", "muscles", "equipment"}
        if unknown:
            raise ValueError(f"entry {number}: unknown keys {sorted(unknown)}")
        exercise_id = raw.get("id")
        if not isinstance(exercise_id, int) or isinstance(exercise_id, bool):
            raise ValueError(f"entry {number}: id must be an integer")
        where = f"entry {number} (id {exercise_id})"
        if exercise_id in entries:
            raise ValueError(f"{where}: duplicate id")
        aliases, muscles = raw.get("aliases", []), raw.get("muscles")
        equipment = raw.get("equipment")
        if not _strings(aliases):
            raise ValueError(f"{where}: aliases must be a list of non-empty strings")
        if not _strings(muscles) or not muscles:
            raise ValueError(f"{where}: muscles must be a non-empty list of strings")
        bad = [m for m in muscles if m not in MUSCLES]
        if bad or len(set(muscles)) != len(muscles):
            raise ValueError(f"{where}: unknown or repeated muscles {bad or muscles}")
        if not isinstance(equipment, str) or not equipment.strip():
            raise ValueError(f"{where}: equipment must be a non-empty string")
        tidied = [normalise(a) for a in aliases]
        if len(set(tidied)) != len(tidied):
            raise ValueError(f"{where}: aliases repeat once tidied: {tidied}")
        entries[exercise_id] = CatalogEntry(
            exercise_id, tuple(tidied), tuple(muscles), equipment.strip()
        )

    if not entries:
        raise ValueError("no entries")
    return entries


def load_catalog(path: Path = CATALOG) -> dict[int, CatalogEntry]:
    with path.open("rb") as file:
        return parse_catalog(tomllib.load(file))


def check_catalog(catalog: Mapping[int, CatalogEntry], names: Mapping[int, str]) -> list[str]:
    """Problems with the catalog against the exercises table (id -> name); empty if none."""
    problems = [f"id {i} is not in the exercises table" for i in sorted(catalog.keys() - names)]
    problems += [f"exercise {i} ({names[i]}) has no catalog entry"
                 for i in sorted(names.keys() - catalog.keys())]
    by_name: dict[str, set[int]] = {}
    for i, name in names.items():
        by_name.setdefault(normalise(name), set()).add(i)
    for entry in catalog.values():
        for alias in entry.aliases:
            owners = by_name.get(alias, set())
            if entry.exercise_id in owners:
                problems.append(f"id {entry.exercise_id}: alias {alias!r} is its own name")
            elif owners:
                problems.append(
                    f"id {entry.exercise_id}: alias {alias!r} is the name of {sorted(owners)}"
                )
    return problems


async def store_catalog(conn: asyncpg.Connection, catalog: Mapping[int, CatalogEntry]) -> None:
    """Replace the catalog's aliases, muscles and equipment, in one transaction."""
    async with conn.transaction():
        await conn.execute("delete from exercise_aliases where source = 'catalog'")
        await conn.execute("delete from exercise_muscles")
        await conn.execute("delete from exercise_details")
        await conn.executemany(
            "insert into exercise_aliases (exercise_id, alias, source) values ($1, $2, 'catalog')",
            [(e.exercise_id, a) for e in catalog.values() for a in e.aliases],
        )
        await conn.executemany(
            "insert into exercise_muscles (exercise_id, muscle, position) values ($1, $2, $3)",
            [(e.exercise_id, m, p) for e in catalog.values() for p, m in enumerate(e.muscles)],
        )
        await conn.executemany(
            "insert into exercise_details (exercise_id, equipment) values ($1, $2)",
            [(e.exercise_id, e.equipment) for e in catalog.values()],
        )
