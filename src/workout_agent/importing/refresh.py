"""Bring the database up to date with Jefit: download, import, re-index.

refresh_from_jefit is what the once-a-day gate runs on the first chat message of the day
(sessions/daily_checks.py). Every step is plain code, no model. Failures are values: a
result with status "unavailable" and a note, never an exception into the chat.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import asyncpg
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.cache import cached
from workout_agent.embeddings.index import COLLECTION, Embedder, index_exercises
from workout_agent.embeddings.openrouter import MODEL_NAME, collection_for
from workout_agent.importing.jefit_export import JefitFetchError, download_export
from workout_agent.importing.loader import build_plan, load_plan
from workout_agent.importing.parsers import parse_exercise_logs
from workout_agent.importing.routine import load_routine, parse_routines
from workout_agent.importing.sections import split_sections
from workout_agent.importing.timed_exercises import TIMED_EXERCISE_IDS
from workout_agent.settings import Settings

logger = logging.getLogger(__name__)

EXPORT_DIR = Path(__file__).parents[3] / "data"

Download = Callable[[Settings, Path], Awaitable[Path]]


@dataclass
class RefreshResult:
    status: Literal["ok", "unavailable"]
    note: str = ""
    logs: int = 0
    exercises_indexed: int = 0


@dataclass
class ImportCounts:
    logs: int
    errors: list[str]


async def load_export(conn: asyncpg.Connection, csv_path: Path) -> ImportCounts:
    """Import one Jefit export file: history only grows, the routine is replaced."""
    sections = split_sections(csv_path.read_text(encoding="utf-8"))
    parsed = parse_exercise_logs(sections["EXERCISE LOGS"])
    plan = build_plan(parsed.rows, timed_exercise_ids=TIMED_EXERCISE_IDS)
    routine = parse_routines(sections["ROUTINES"])
    await load_plan(conn, plan)
    await load_routine(conn, routine)
    return ImportCounts(len(parsed.rows), parsed.errors + plan.errors + routine.errors)


async def _download(settings: Settings, dest_dir: Path) -> Path:
    assert settings.jefit_email and settings.jefit_password  # checked by the caller
    return await download_export(
        settings.jefit_email, settings.jefit_password.get_secret_value(), dest_dir
    )


async def refresh_from_jefit(
    settings: Settings,
    conn: asyncpg.Connection,
    qdrant: AsyncQdrantClient | None,
    embedder: Embedder | None,
    download: Download = _download,
    export_dir: Path = EXPORT_DIR,
) -> RefreshResult:
    """Download the export, import it, and re-index exercises (when search is set up)."""
    if not (settings.jefit_email and settings.jefit_password):
        return RefreshResult("unavailable", "JEFIT_EMAIL and JEFIT_PASSWORD are not set")
    try:
        path = await download(settings, export_dir)
    except JefitFetchError as error:
        logger.warning("Jefit download failed: %s", error)
        return RefreshResult("unavailable", str(error))
    try:
        counts = await load_export(conn, path)
        indexed = 0
        if qdrant is not None and embedder is not None:
            collection = collection_for(MODEL_NAME, COLLECTION)
            cache = cached(conn, MODEL_NAME, embedder)  # unchanged names cost nothing
            indexed = (await index_exercises(conn, qdrant, cache, collection)).exercises
    except Exception as error:  # a bad file or a down service is a value, not a crash
        logger.exception("Jefit import failed for %s", path.name)
        return RefreshResult("unavailable", f"importing {path.name} failed: {error}")
    return RefreshResult("ok", path.name, counts.logs, indexed)
