"""Check config/exercise_catalog.toml against the database and store it.

Usage: uv run python scripts/load_catalog.py
Run after scripts/load_jefit.py and before scripts/index_exercises.py. Prints every problem
and stores nothing if there is one. Safe to run again: it replaces the catalog rows.
"""

import asyncio
import sys

import asyncpg

from workout_agent.catalog import CATALOG, check_catalog, load_catalog, store_catalog
from workout_agent.db.migrate import apply_migrations
from workout_agent.settings import Settings


async def main() -> int:
    catalog = load_catalog()
    conn = await asyncpg.connect(Settings().database_url)
    try:
        await apply_migrations(conn)
        names = {r["exercise_id"]: r["name"] for r in await conn.fetch("select * from exercises")}
        problems = check_catalog(catalog, names)
        if problems:
            print(f"{CATALOG.name}: {len(problems)} problem(s), nothing stored")
            for problem in problems:
                print(f"  {problem}")
            return 1
        await store_catalog(conn, catalog)
        aliases = await conn.fetchval("select count(*) from exercise_aliases")
        muscles = await conn.fetchval("select count(*) from exercise_muscles")
    finally:
        await conn.close()

    print(f"exercises: {len(catalog)}  aliases: {aliases}  muscle rows: {muscles}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
