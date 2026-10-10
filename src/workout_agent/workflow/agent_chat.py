"""Chat with the agent: one message in, one reply out, with memory of the conversation.

The agent decides which tools to call. We keep every message of the conversation in
AgentSession.history and give it back on the next turn, so "the incline one" makes sense.
"""

from dataclasses import dataclass, field

import asyncpg
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from workout_agent.agents.workout_agent import AgentDeps
from workout_agent.tools.find_exercise import Searcher


@dataclass
class AgentSession:
    history: list[ModelMessage] = field(default_factory=list)


async def handle_agent_message(
    conn: asyncpg.Connection,
    session: AgentSession,
    text: str,
    agent: Agent[AgentDeps, str],
    search: Searcher | None = None,
) -> str:
    deps = AgentDeps(conn=conn, search=search)  # fresh evidence list for each question
    result = await agent.run(text, deps=deps, message_history=session.history)
    # all_messages() already includes the old history, so replace it, don't extend it.
    session.history = result.all_messages()
    return result.output