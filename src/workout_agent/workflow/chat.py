"""The chat workflow: one message in, one reply out.

A model-or-rules "router" decides what the user meant. Everything after that is
plain code: run the tool, write the reply from a template, verify its numbers.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import asyncpg

from workout_agent.tools.find_exercise import ExerciseMatch, FindResult, find_exercise
from workout_agent.tools.get_last_session import get_last_session
from workout_agent.workflow.replies import (
    HELP_TEXT,
    UNVERIFIED_REPLY,
    format_find_question,
    format_last_session,
    verify_reply,
)


@dataclass
class RouteDecision:
    """What the router thinks the user meant. The router never answers anything."""

    intent: Literal["last_session", "pick_option", "unknown"]
    exercise_text: str | None = None  # "bench press", for last_session
    option_number: int | None = None  # 1, 2, ..., for pick_option


@dataclass
class Session:
    """What we remember between messages in one conversation."""

    pending: list[ExerciseMatch] | None = None  # options we asked the user to pick from
    pending_query: str = ""  # what they originally typed, so we can re-ask


# A router takes (message text, "is a question pending?") and returns a decision.
Route = Callable[[str, bool], Awaitable[RouteDecision]]


async def _answer(conn: asyncpg.Connection, exercise_id: int) -> str:
    """Fetch the last session, write the reply, and check it before returning it."""
    result = await get_last_session(conn, exercise_id)
    reply = format_last_session(result)

    # Every number in the reply must come from the tool result. If not, say nothing.
    if not verify_reply(reply, result):
        return UNVERIFIED_REPLY
    return reply


async def handle_message(
    conn: asyncpg.Connection,
    session: Session,
    text: str,
    route: Route,
) -> str:
    # Step 1: ask the router what this message means.
    decision = await route(text, session.pending is not None)

    # Case A: the user is answering our earlier "which one?" question.
    if decision.intent == "pick_option":
        if session.pending is None:
            # They typed "1" but we never asked anything.
            return "Tell me which exercise you mean, e.g. 'last time I did bench press'."

        number = decision.option_number
        if number is None or not 1 <= number <= len(session.pending):
            # Not a valid choice: ask again, and keep the options.
            question = FindResult(status="ambiguous", matches=session.pending)
            return format_find_question(question, session.pending_query)

        # Valid choice: take that exercise and forget the question.
        chosen = session.pending[number - 1]
        session.pending = None
        session.pending_query = ""
        return await _answer(conn, chosen.exercise_id)

    # Case B: the user asked about an exercise.
    if decision.intent == "last_session" and decision.exercise_text:
        result = await find_exercise(conn, decision.exercise_text)

        if result.status == "no_rows":
            return format_find_question(result, decision.exercise_text)

        if result.status == "ambiguous":
            # Remember the options and what was asked, then ask the user.
            session.pending = result.matches
            session.pending_query = decision.exercise_text
            return format_find_question(result, decision.exercise_text)

        # status "ok": exactly one exercise, so answer straight away.
        return await _answer(conn, result.matches[0].exercise_id)

    # Case C: unknown intent, or "last_session" with no exercise text.
    return HELP_TEXT