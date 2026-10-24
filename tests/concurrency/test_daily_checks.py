"""Two chats sending their first message of the day at the same moment."""

import asyncio
from datetime import date

import asyncpg

from workout_agent.db.migrate import apply_migrations
from workout_agent.sessions.daily_checks import latest_record_today
from workout_agent.settings import Settings

TEST_DB = "workout_test"  # the database tests/integration/conftest.py creates


async def test_two_first_messages_at_once_leave_one_stored_row():
    url = Settings().database_url.rsplit("/", 1)[0] + "/" + TEST_DB
    conns = [await asyncpg.connect(url) for _ in range(3)]
    try:
        await apply_migrations(conns[0])
        await conns[0].execute("truncate daily_checks, sets, exercise_logs, exercises cascade")
        await conns[0].execute("insert into exercises (exercise_id, name) values (1, 'Squat')")
        await conns[0].execute(
            "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
            "values (1, 1, '2026-02-09')"
        )

        today = date.today()
        first, second = await asyncio.gather(
            latest_record_today(conns[1], today), latest_record_today(conns[2], today)
        )

        assert first[0] == second[0] == {"status": "ok", "latest_on": "2026-02-09"}
        assert await conns[0].fetchval("select count(*) from daily_checks") == 1
    finally:
        for c in conns:
            await c.close()
