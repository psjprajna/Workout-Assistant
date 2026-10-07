import asyncio
import logging
import sys
from pathlib import Path

import asyncpg

from workout_agent.db.migrate import apply_migrations
from workout_agent.importing.loader import build_plan, load_plan
from workout_agent.importing.parsers import parse_exercise_logs
from workout_agent.importing.sections import split_sections
from workout_agent.importing.timed_exercises import TIMED_EXERCISE_IDS
from workout_agent.settings import Settings


async def main(csv_path: str) -> None:
    text = Path(csv_path).read_text(encoding="utf-8")
    sections = split_sections(text)

    parsed = parse_exercise_logs(sections["EXERCISE LOGS"])
    plan = build_plan(parsed.rows, timed_exercise_ids=TIMED_EXERCISE_IDS)

    print(f"parsed logs:   {len(parsed.rows)} ({len(parsed.errors)} row errors)")
    print(f"plan: {len(plan.exercises)} exercises, {len(plan.logs)} logs, "
          f"{len(plan.sets)} sets, {len(plan.errors)} bad logs")
    for error in (parsed.errors + plan.errors)[:20]:
        print("  !", error)

    conn = await asyncpg.connect(Settings().database_url)
    try:
        await apply_migrations(conn)
        await load_plan(conn, plan)
        for table in ("exercises", "exercise_logs", "sets"):
            count = await conn.fetchval(f"select count(*) from {table}")
            print(f"db {table}: {count}")
    finally:
        await conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(sys.argv[1]))