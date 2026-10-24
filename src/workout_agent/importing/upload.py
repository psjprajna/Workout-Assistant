"""Import a Jefit export the user uploads in the chat, then re-index exercises.

import_export is what the API runs when the chat page sends a file. Plain code, no model.
Failures are values: a result with status "rejected" and a note the page can show, never
an exception into the chat. A file that isn't a Jefit export is rejected before anything
is written, so a wrong file can't wipe the stored plan.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import asyncpg
from qdrant_client import AsyncQdrantClient

from workout_agent.embeddings.cache import cached
from workout_agent.embeddings.index import COLLECTION, Embedder, index_exercises
from workout_agent.embeddings.openrouter import MODEL_NAME, collection_for
from workout_agent.importing.loader import build_plan, load_plan
from workout_agent.importing.parsers import parse_exercise_logs
from workout_agent.importing.routine import load_routine, parse_routines
from workout_agent.importing.sections import split_sections
from workout_agent.importing.timed_exercises import TIMED_EXERCISE_IDS

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    status: Literal["ok", "rejected"]
    note: str = ""
    logs: int = 0
    latest_on: str | None = None  # newest logged session after the import, ISO date
    exercises_indexed: int = 0


async def import_export(
    conn: asyncpg.Connection,
    text: str,
    qdrant: AsyncQdrantClient | None = None,
    embedder: Embedder | None = None,
) -> ImportResult:
    """Import one export: history only grows, the routine is replaced, exercises re-indexed."""
    sections = split_sections(text)
    missing = [name for name in ("EXERCISE LOGS", "ROUTINES") if name not in sections]
    if missing:
        return ImportResult("rejected", f"not a Jefit export: no {', '.join(missing)} section")
    parsed = parse_exercise_logs(sections["EXERCISE LOGS"])
    routine = parse_routines(sections["ROUTINES"])
    if not parsed.rows or not routine.days:
        return ImportResult("rejected", "the file has no logged sessions or no routine days")
    plan = build_plan(parsed.rows, timed_exercise_ids=TIMED_EXERCISE_IDS)
    try:
        await load_plan(conn, plan)
        await load_routine(conn, routine)
        indexed = 0
        if qdrant is not None and embedder is not None:
            cache = cached(conn, MODEL_NAME, embedder)  # unchanged names cost nothing
            collection = collection_for(MODEL_NAME, COLLECTION)
            indexed = (await index_exercises(conn, qdrant, cache, collection)).exercises
    except Exception as error:  # a bad row or a down service is a value, not a crash
        logger.exception("export import failed")
        return ImportResult("rejected", f"importing failed: {error}")
    latest = await conn.fetchval("SELECT max(performed_on) FROM exercise_logs")
    problems = len(parsed.errors) + len(plan.errors) + len(routine.errors)
    note = f"{problems} rows skipped" if problems else ""
    return ImportResult("ok", note, len(parsed.rows), latest.isoformat() if latest else None, indexed)
