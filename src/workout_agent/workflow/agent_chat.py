"""Chat with the agent: one message in, progress events and one reply out, with memory.

The agent decides which tools to call. stream_agent_message reports each tool call as it
starts and ends, then the reply. Each answered turn is saved in Postgres
(sessions/chat_store.py), and the last few turns go back to the model on the next one,
so "the incline one" makes sense, even after a restart. Before a reply is shown, code
checks every number in it against the tool results the model saw (this turn's, and those
in the history it was given); a reply that fails is replaced by a refusal and not saved.
A failed or abandoned run saves nothing. handle_agent_message is the same path for
callers that only want the reply.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import asyncpg
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
)
from pydantic_ai.usage import UsageLimits

from workout_agent.agents.workout_agent import AgentDeps
from workout_agent.sessions.chat_store import load_history, save_turn, try_lock_chat, unlock_chat
from workout_agent.sessions.history import shrink_old_tool_results
from workout_agent.sessions.titles import chat_title
from workout_agent.tools.find_exercise import Searcher
from workout_agent.workflow.events import ChatEvent, Error, Final, ToolEnd, ToolStart, tool_label
from workout_agent.workflow.replies import UNVERIFIED_REPLY, verify_agent_reply

BUSY_REPLY = "This chat is busy answering another message. Please send yours again."
MAX_TOOL_CALLS = 6  # docs/amendments.md §9


def _tool_status(event: FunctionToolResultEvent) -> str:
    part = event.part
    if isinstance(part, ToolReturnPart) and isinstance(part.content, dict):
        return str(part.content.get("status", "ok"))
    return "retry"  # the model sent bad arguments and is asked to try again


def _tool_results_in(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """The tool results in earlier turns, as the model is shown them (old ones shrunk)."""
    return [
        cast(dict[str, Any], part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and isinstance(part.content, dict)
    ]


async def _run_agent(
    agent: Agent[AgentDeps, str],
    text: str,
    deps: AgentDeps,
    history: list[ModelMessage],
    queue: asyncio.Queue[ChatEvent | None],
) -> AgentRunResult[str]:
    """Run the agent, putting a ToolStart/ToolEnd on the queue per tool call, then None."""
    try:
        # iter() walks the run node by node. Only the tool-calling nodes are streamed,
        # for their call/result events; the model calls stay plain (not streamed) requests.
        async with agent.iter(
            text,
            deps=deps,
            message_history=history,
            usage_limits=UsageLimits(tool_calls_limit=MAX_TOOL_CALLS),
        ) as run:
            async for node in run:
                if not Agent.is_call_tools_node(node):
                    continue
                async with node.stream(run.ctx) as events:
                    async for event in events:
                        if isinstance(event, FunctionToolCallEvent):
                            tool = event.part.tool_name
                            label = tool_label(tool, event.part.args_as_dict())
                            queue.put_nowait(ToolStart(tool=tool, label=label))
                        elif isinstance(event, FunctionToolResultEvent):
                            tool = event.part.tool_name or ""
                            queue.put_nowait(ToolEnd(tool=tool, status=_tool_status(event)))
    finally:
        queue.put_nowait(None)
    if run.result is None:  # iter() always ends with a result unless it raised
        raise RuntimeError("agent run ended without a result")
    return run.result


async def stream_agent_message(
    conn: asyncpg.Connection,
    chat_id: str,
    text: str,
    agent: Agent[AgentDeps, str],
    search: Searcher | None = None,
    history_turns: int = 10,
    full_tool_turns: int = 2,
) -> AsyncGenerator[ChatEvent]:
    # Turn a second message away before any work, not after the user watched it run.
    if not await try_lock_chat(conn, chat_id):
        yield Error(message=BUSY_REPLY)
        return
    try:
        history, last_turn = await load_history(conn, chat_id, history_turns)
        history = shrink_old_tool_results(history, full_tool_turns)
        deps = AgentDeps(conn=conn, search=search)  # fresh evidence list for each question
        # The run happens in its own task and hands events over through a queue: the
        # agent's internals must start and finish in one task, and this generator may
        # be closed early (the client went away), which then just cancels that task.
        queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()
        task = asyncio.create_task(_run_agent(agent, text, deps, history, queue))
        try:
            while (event := await queue.get()) is not None:
                yield event
            result = await task  # raises if the run failed
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task})  # it must stop using conn before we go on

        # Verification is code: every number must come from a tool result the model saw.
        seen = _tool_results_in(history) + deps.evidence
        if not verify_agent_reply(result.output, seen):
            yield Final(reply=UNVERIFIED_REPLY, verified=False, checked=len(seen))
            return  # the request did not succeed, so nothing is saved

        title, is_fallback = chat_title(text, deps.evidence)
        try:
            # new_messages() is only this turn; older turns are already saved.
            await save_turn(
                conn, chat_id, last_turn + 1, text, result.output,
                result.new_messages(), deps.evidence, title, is_fallback,
            )
        except asyncpg.UniqueViolationError:
            # Another message in this chat finished first; don't overwrite its turn.
            yield Error(message=BUSY_REPLY)
            return
        yield Final(reply=result.output, verified=True, checked=len(seen))
    finally:
        await unlock_chat(conn, chat_id)


async def handle_agent_message(
    conn: asyncpg.Connection,
    chat_id: str,
    text: str,
    agent: Agent[AgentDeps, str],
    search: Searcher | None = None,
    history_turns: int = 10,
    full_tool_turns: int = 2,
) -> str:
    """The reply only: runs the same stream and keeps its last event."""
    reply = ""
    async for event in stream_agent_message(
        conn, chat_id, text, agent, search, history_turns, full_tool_turns
    ):
        if isinstance(event, Final):
            reply = event.reply
        elif isinstance(event, Error):
            reply = event.message
    return reply
