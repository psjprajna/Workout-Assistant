"""The once-a-day Jefit refresh: the first chat message of the day downloads and imports
the export; later chats that day (and failures) do not run it again. The download and the
import are replaced by fakes; the browser and the real file are not needed.
"""

import asyncio
import json
from datetime import date
from pathlib import Path

import asyncpg
from pydantic import SecretStr
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import workout_agent.importing.refresh as refresh_module
from workout_agent.agents.workout_agent import build_agent
from workout_agent.importing.jefit_export import JefitFetchError
from workout_agent.importing.refresh import ImportCounts, RefreshResult, refresh_from_jefit
from workout_agent.sessions.daily_checks import refresh_today
from workout_agent.settings import Settings
from workout_agent.workflow.agent_chat import stream_agent_message
from workout_agent.workflow.events import Final, ToolEnd, ToolStart

AGENT = build_agent(Settings(openrouter_api_key=SecretStr("test")))
TODAY = date.today()
WITH_LOGIN = Settings(jefit_email="me@example.com", jefit_password=SecretStr("pw"))


def help_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart("final_result_Help", {})])


class FakeRefresh:
    """Stands in for the app's refresh; counts runs and can be made to fail."""

    def __init__(self, result: RefreshResult | None = None) -> None:
        self.runs = 0
        self.result = result or RefreshResult("ok", "export.csv", logs=3)

    async def __call__(self, conn) -> dict:
        self.runs += 1
        return {"status": self.result.status, "note": self.result.note,
                "logs": self.result.logs, "exercises_indexed": 0}


async def say(conn, chat_id: str, refresh) -> list:
    with AGENT.override(model=FunctionModel(help_model)):
        return [
            e async for e in stream_agent_message(conn, chat_id, "hello", AGENT, refresh=refresh)
        ]


def steps(events: list) -> list[str]:
    return [e.tool for e in events if isinstance(e, ToolStart)]


async def test_the_first_message_of_the_day_refreshes_before_looking_up(conn):
    async def refresh(conn):
        await conn.execute("insert into exercises (exercise_id, name) values (1, 'Squat')")
        await conn.execute(
            "insert into exercise_logs (exercise_log_id, exercise_id, performed_on) "
            "values (1, 1, $1)",
            TODAY,
        )
        return {"status": "ok", "note": "export.csv", "logs": 1, "exercises_indexed": 0}

    events = await say(conn, "t1", refresh)

    assert steps(events) == ["jefit_refresh", "latest_record"]
    assert ToolEnd(tool="jefit_refresh", status="ok", note="export.csv") in events
    # The freshness lookup came after the import, so it sees today's session.
    latest = await conn.fetchval(
        "select result from daily_checks where check_name = 'latest_record'"
    )
    assert json.loads(latest)["latest_on"] == TODAY.isoformat()
    assert isinstance(events[-1], Final)


async def test_later_chats_the_same_day_do_not_refresh(conn):
    refresh = FakeRefresh()
    await say(conn, "t1", refresh)

    other_chat = await say(conn, "t2", refresh)
    same_chat = await say(conn, "t1", refresh)

    assert refresh.runs == 1
    assert "jefit_refresh" not in steps(other_chat) + steps(same_chat)
    stored = await conn.fetchval(
        "select count(*) from daily_checks where check_name = 'jefit_refresh'"
    )
    assert stored == 1


async def test_a_failed_refresh_is_shown_stored_and_not_retried_that_day(conn):
    refresh = FakeRefresh(RefreshResult("unavailable", "signing in: wrong password?"))

    first = await say(conn, "t1", refresh)
    second = await say(conn, "t2", refresh)

    assert ToolEnd(
        tool="jefit_refresh", status="unavailable", note="signing in: wrong password?"
    ) in first
    assert isinstance(first[-1], Final) and isinstance(second[-1], Final)  # still answered
    assert refresh.runs == 1


async def test_a_refresh_that_raises_does_not_block_the_answer(conn):
    async def broken(conn):
        raise RuntimeError("boom")

    events = await say(conn, "t1", broken)

    assert "jefit_refresh" not in steps(events)
    assert isinstance(events[-1], Final) and events[-1].verified


async def test_no_refresh_given_means_no_step(conn):
    assert "jefit_refresh" not in steps(await say(conn, "t1", None))


async def test_another_chat_refreshing_means_this_one_goes_on(conn, test_database_url):
    other = await asyncpg.connect(test_database_url)
    try:
        assert await other.fetchval("select pg_try_advisory_lock($1)", 7_204_202)
        refresh = FakeRefresh()
        result, ran = await refresh_today(conn, TODAY, lambda: refresh(conn))
    finally:
        await other.close()

    assert (result, ran) == (None, False) and refresh.runs == 0


def fake_download(path: Path | Exception):
    async def download(settings, dest_dir):
        if isinstance(path, Exception):
            raise path
        return path

    return download


async def test_refresh_without_a_login_is_unavailable(conn):
    result = await refresh_from_jefit(Settings(jefit_email=None), conn, None, None)

    assert result.status == "unavailable" and "JEFIT_EMAIL" in result.note


async def test_refresh_reports_a_failed_download(conn):
    download = fake_download(JefitFetchError("signing in: wrong password?"))

    result = await refresh_from_jefit(WITH_LOGIN, conn, None, None, download=download)

    assert result == RefreshResult("unavailable", "signing in: wrong password?")


async def test_refresh_downloads_then_imports(conn, monkeypatch, tmp_path):
    seen = []

    async def fake_load(conn, path):
        seen.append(path.name)
        return ImportCounts(7, [])

    monkeypatch.setattr(refresh_module, "load_export", fake_load)
    export = tmp_path / "me_20261024.csv"
    export.write_text("### SETTING")
    download = fake_download(export)

    result = await refresh_from_jefit(WITH_LOGIN, conn, None, None, download=download)

    assert result == RefreshResult("ok", "me_20261024.csv", logs=7) and seen == ["me_20261024.csv"]


async def test_refresh_reports_an_import_that_fails(conn, monkeypatch, tmp_path):
    async def bad_load(conn, path):
        raise KeyError("EXERCISE LOGS")

    monkeypatch.setattr(refresh_module, "load_export", bad_load)
    export = tmp_path / "x.csv"
    export.write_text("junk")
    download = fake_download(export)

    result = await refresh_from_jefit(WITH_LOGIN, conn, None, None, download=download)

    assert result.status == "unavailable" and "x.csv" in result.note


async def test_two_chats_at_once_refresh_once(conn, test_database_url):
    other = await asyncpg.connect(test_database_url)
    started = asyncio.Event()
    release = asyncio.Event()
    runs = 0

    async def slow():
        nonlocal runs
        runs += 1
        started.set()
        await release.wait()
        return {"status": "ok", "note": "", "logs": 0, "exercises_indexed": 0}

    try:
        first = asyncio.create_task(refresh_today(conn, TODAY, slow))
        await started.wait()
        second = await refresh_today(other, TODAY, slow)  # while the first is running
        release.set()
        await first
    finally:
        await other.close()

    assert second == (None, False) and runs == 1
