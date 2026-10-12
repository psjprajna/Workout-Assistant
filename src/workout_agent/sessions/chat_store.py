"""Chat history in Postgres: the chats table and one append-only row per answered turn.

The agent path loads the last few turns' model messages before a run and saves the new
turn after it succeeds. The /chats routes read, rename and delete through here.
A turn's number comes from the history load; if two requests for one chat race, the
second insert of the same (chat_id, turn_no) fails instead of overwriting anything.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter


@dataclass
class ChatSummary:
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass
class TurnView:
    user_text: str
    reply_text: str
    created_at: datetime


@dataclass
class ChatDetail:
    chat_id: str
    title: str
    created_at: datetime
    turns: list[TurnView]


async def load_history(
    conn: asyncpg.Connection, chat_id: str, last_turns: int
) -> tuple[list[ModelMessage], int]:
    """The model messages of the last `last_turns` turns, oldest first, and the last turn_no."""
    rows = await conn.fetch(
        "select messages from chat_turns where chat_id = $1 "
        "order by turn_no desc limit $2",
        chat_id,
        last_turns,
    )
    history: list[ModelMessage] = []
    for row in reversed(rows):
        history.extend(ModelMessagesTypeAdapter.validate_json(row["messages"]))
    last = await conn.fetchval(
        "select coalesce(max(turn_no), 0) from chat_turns where chat_id = $1", chat_id
    )
    return history, int(last or 0)


async def save_turn(
    conn: asyncpg.Connection,
    chat_id: str,
    turn_no: int,
    user_text: str,
    reply_text: str,
    new_messages: list[ModelMessage],
    evidence: list[dict[str, Any]],
    title: str,
    title_is_fallback: bool,
) -> None:
    """Create or touch the chat, then append the turn. Raises UniqueViolationError on a race."""
    async with conn.transaction():
        # A fallback title (the first message) gives way to a real one; a real one stays.
        await conn.execute(
            """
            insert into chats (chat_id, title, title_is_fallback) values ($1, $2, $3)
            on conflict (chat_id) do update set
                updated_at = now(),
                title = case when chats.title_is_fallback and not excluded.title_is_fallback
                             then excluded.title else chats.title end,
                title_is_fallback = chats.title_is_fallback and excluded.title_is_fallback
            """,
            chat_id,
            title,
            title_is_fallback,
        )
        await conn.execute(
            "insert into chat_turns "
            "(chat_id, turn_no, user_text, reply_text, messages, evidence) "
            "values ($1, $2, $3, $4, $5::jsonb, $6::jsonb)",
            chat_id,
            turn_no,
            user_text,
            reply_text,
            ModelMessagesTypeAdapter.dump_json(new_messages).decode(),
            json.dumps(evidence),
        )


async def list_chats(
    conn: asyncpg.Connection, limit: int, before: datetime | None = None
) -> list[ChatSummary]:
    """Most recently used chats first; `before` is the updated_at of the previous page's last."""
    rows = await conn.fetch(
        "select chat_id, title, created_at, updated_at from chats "
        "where $2::timestamptz is null or updated_at < $2 "
        "order by updated_at desc limit $1",
        limit,
        before,
    )
    return [ChatSummary(**dict(r)) for r in rows]


async def get_chat(conn: asyncpg.Connection, chat_id: str) -> ChatDetail | None:
    chat = await conn.fetchrow(
        "select chat_id, title, created_at from chats where chat_id = $1", chat_id
    )
    if chat is None:
        return None
    turns = await conn.fetch(
        "select user_text, reply_text, created_at from chat_turns "
        "where chat_id = $1 order by turn_no",
        chat_id,
    )
    return ChatDetail(**dict(chat), turns=[TurnView(**dict(t)) for t in turns])


async def delete_chat(conn: asyncpg.Connection, chat_id: str) -> None:
    await conn.execute("delete from chats where chat_id = $1", chat_id)  # turns cascade


async def rename_chat(conn: asyncpg.Connection, chat_id: str, title: str) -> bool:
    """Set a title the user chose; later turns never replace it. False if no such chat."""
    status = await conn.execute(
        "update chats set title = $2, title_is_fallback = false where chat_id = $1",
        chat_id,
        title,
    )
    return status == "UPDATE 1"
