"""Uploading a newer Jefit export in the chat: the page asks once a day, the API imports
the file it is sent, and a file that isn't a Jefit export is turned away without touching
the stored data.
"""

from datetime import date, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI

from workout_agent.api.app import create_app
from workout_agent.sessions import daily_checks
from workout_agent.ui.client import export_wanted, upload_export
from workout_agent.workflow.simple_router import simple_route

TODAY = date.today()
MINI = Path(__file__).parents[1] / "fixtures" / "mini_jefit.csv"
ROUTINE = """### ROUTINES ##########################################

row_id,_id,name,dayaweek
1,1,PPL,4
row_id,_id,package,name,dayIndex,rest_day
1,1,1,"Push A",1,0
row_id,_id,belongplan,exercise_id,exercisename,mysort,targetrep,setcount
1,1,1,10,"Bench Press, Flat",0,"8,8",2
######################################################
"""


def export_text(extra: str = ROUTINE) -> str:
    return MINI.read_text(encoding="utf-8") + "\n" + extra


def client_for(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_the_page_is_asked_to_prompt_once_a_day(conn):
    first = await daily_checks.ask_for_export_today(conn, TODAY)
    second = await daily_checks.ask_for_export_today(conn, TODAY)
    next_day = await daily_checks.ask_for_export_today(conn, TODAY + timedelta(days=1))

    assert first == (True, None) and second == (False, None) and next_day == (True, None)


async def test_no_prompt_when_the_newest_session_is_today(conn):
    await conn.execute("insert into exercises (exercise_id, name) values (1, 'Squat')")
    await conn.execute(
        "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) values (1, 1, $1)",
        TODAY,
    )

    assert await daily_checks.ask_for_export_today(conn, TODAY) == (False, TODAY)


async def test_an_uploaded_export_is_imported(conn, test_database_url):
    app = create_app(simple_route, test_database_url)
    async with app.router.lifespan_context(app), client_for(app) as client:
        message = await upload_export(client, export_text().encode())

    assert message.startswith("Imported 2 logged sessions; your data now runs through 2026-01-05")
    assert await conn.fetchval("select count(*) from exercise_logs") == 2
    assert await conn.fetchval("select count(*) from routine_days") == 1


async def test_an_import_makes_the_freshness_check_look_again(conn, test_database_url):
    await conn.execute(
        "insert into daily_checks (check_name, checked_on, result) values "
        "('latest_record', $1, '{\"status\": \"no_rows\", \"latest_on\": null}')",
        TODAY,
    )
    app = create_app(simple_route, test_database_url)
    async with app.router.lifespan_context(app), client_for(app) as client:
        await upload_export(client, export_text().encode())

    assert await conn.fetchval("select count(*) from daily_checks") == 0


async def test_a_file_that_is_not_an_export_changes_nothing(conn, test_database_url):
    await conn.execute("insert into routines values (1, 'Mine')")
    app = create_app(simple_route, test_database_url)
    async with app.router.lifespan_context(app), client_for(app) as client:
        junk = await upload_export(client, b"a,b\n1,2\n")
        no_plan = await upload_export(client, export_text(extra="").encode())
        binary = await upload_export(client, b"\xff\xfe\x00")

    assert junk.startswith("That file was not imported: not a Jefit export")
    assert no_plan.startswith("That file was not imported:")
    assert binary.startswith("That file was not imported")
    assert await conn.fetchval("select name from routines") == "Mine"
    assert await conn.fetchval("select count(*) from exercise_logs") == 0


async def test_prompt_endpoint_asks_first_and_not_again(conn, test_database_url):
    app = create_app(simple_route, test_database_url)
    async with app.router.lifespan_context(app), client_for(app) as client:
        first = await export_wanted(client)
        second = await export_wanted(client)

    assert first == "none yet" and second is None


async def test_prompt_is_skipped_when_the_api_is_down():
    async def refuse(request):
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://test")

    assert await export_wanted(client) is None
