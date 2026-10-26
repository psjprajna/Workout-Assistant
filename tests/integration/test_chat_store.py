"""Integration tests: chat history in Postgres (sessions/chat_store.py)."""

import asyncpg
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from workout_agent.sessions import chat_store


def _turn_messages(n: int) -> list:
    return [
        ModelRequest(parts=[UserPromptPart(content=f"question {n}")]),
        ModelResponse(parts=[TextPart(content=f"answer {n}")]),
    ]


async def _save(conn, chat_id, n, title="question", fallback=True):
    await chat_store.save_turn(
        conn, chat_id, n, f"question {n}", f"answer {n}", _turn_messages(n),
        [{"status": "ok", "n": n}], title, fallback,
    )


def _prompts(history) -> list[str]:
    return [
        str(p.content) for m in history if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, UserPromptPart)
    ]


async def test_new_chat_has_no_history(conn):
    assert await chat_store.load_history(conn, "new", 10) == ([], 0)


async def test_turns_come_back_in_order(conn):
    for n in (1, 2, 3):
        await _save(conn, "c", n)

    history, last = await chat_store.load_history(conn, "c", 10)

    assert last == 3
    assert _prompts(history) == ["question 1", "question 2", "question 3"]
    assert isinstance(history[-1], ModelResponse)


async def test_only_the_last_turns_are_loaded_but_numbering_continues(conn):
    for n in (1, 2, 3):
        await _save(conn, "c", n)

    history, last = await chat_store.load_history(conn, "c", 2)

    assert _prompts(history) == ["question 2", "question 3"]
    assert last == 3


async def test_evidence_is_stored(conn):
    await _save(conn, "c", 1)
    stored = await conn.fetchval("select evidence::text from chat_turns where chat_id = 'c'")
    assert '"n": 1' in stored


async def test_same_turn_twice_is_rejected_and_nothing_is_overwritten(conn):
    await _save(conn, "c", 1)
    with pytest.raises(asyncpg.UniqueViolationError):
        await chat_store.save_turn(
            conn, "c", 1, "other", "other", _turn_messages(9), [], "x", True
        )
    chat = await chat_store.get_chat(conn, "c")
    assert chat is not None
    assert [t.user_text for t in chat.turns] == ["question 1"]


async def test_chats_do_not_see_each_other(conn):
    await _save(conn, "a", 1)
    await _save(conn, "b", 1)
    await _save(conn, "b", 2)

    history, last = await chat_store.load_history(conn, "a", 10)
    assert last == 1
    assert _prompts(history) == ["question 1"]


async def test_fallback_title_is_upgraded_and_real_title_is_kept(conn):
    await _save(conn, "c", 1, title="last time bench?", fallback=True)
    await _save(conn, "c", 2, title="Barbell Bench Press – last session", fallback=False)
    await _save(conn, "c", 3, title="Barbell Curl", fallback=False)
    await _save(conn, "c", 4, title="hi", fallback=True)

    chat = await chat_store.get_chat(conn, "c")
    assert chat is not None
    assert chat.title == "Barbell Bench Press – last session"


async def test_rename_is_never_replaced(conn):
    await _save(conn, "c", 1, title="hi", fallback=True)
    assert await chat_store.rename_chat(conn, "c", "Chest day")
    await _save(conn, "c", 2, title="Barbell Curl", fallback=False)

    chat = await chat_store.get_chat(conn, "c")
    assert chat is not None
    assert chat.title == "Chest day"


async def test_rename_unknown_chat_is_false(conn):
    assert not await chat_store.rename_chat(conn, "nope", "x")


async def test_list_is_newest_first_and_pages(conn):
    for chat_id in ("a", "b", "c"):
        await _save(conn, chat_id, 1, title=chat_id)

    first = await chat_store.list_chats(conn, 2)
    assert [c.chat_id for c in first] == ["c", "b"]
    rest = await chat_store.list_chats(conn, 2, before=first[-1].updated_at)
    assert [c.chat_id for c in rest] == ["a"]


async def test_delete_removes_the_turns(conn):
    await _save(conn, "c", 1)
    await chat_store.delete_chat(conn, "c")
    assert await chat_store.get_chat(conn, "c") is None
    assert await conn.fetchval("select count(*) from chat_turns where chat_id = 'c'") == 0
