"""Chat with the agent: one message in, one reply out, with memory of the conversation.

The agent decides which tools to call. Each answered turn is saved in Postgres
(sessions/chat_store.py), and the last few turns go back to the model on the next one,
so "the incline one" makes sense, even after a restart. A failed run saves nothing.
"""

import asyncpg
from pydantic_ai import Agent

from workout_agent.agents.workout_agent import AgentDeps
from workout_agent.sessions.chat_store import load_history, save_turn
from workout_agent.sessions.history import shrink_old_tool_results
from workout_agent.sessions.titles import chat_title
from workout_agent.tools.find_exercise import Searcher

BUSY_REPLY = "This chat is busy answering another message. Please send yours again."


async def handle_agent_message(
    conn: asyncpg.Connection,
    chat_id: str,
    text: str,
    agent: Agent[AgentDeps, str],
    search: Searcher | None = None,
    history_turns: int = 10,
    full_tool_turns: int = 2,
) -> str:
    history, last_turn = await load_history(conn, chat_id, history_turns)
    history = shrink_old_tool_results(history, full_tool_turns)
    deps = AgentDeps(conn=conn, search=search)  # fresh evidence list for each question
    result = await agent.run(text, deps=deps, message_history=history)
    title, is_fallback = chat_title(text, deps.evidence)
    try:
        # new_messages() is only this turn; older turns are already saved.
        await save_turn(
            conn, chat_id, last_turn + 1, text, result.output,
            result.new_messages(), deps.evidence, title, is_fallback,
        )
    except asyncpg.UniqueViolationError:
        # Another message in this chat finished first; don't overwrite its turn.
        return BUSY_REPLY
    return result.output
