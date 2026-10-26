"""Chat with the agent: one message in, progress events and one reply out, with memory.

The agent decides which tools to call. stream_agent_message reports each tool call as it
starts and ends, then the reply. Each answered turn is saved in Postgres
(sessions/chat_store.py), and the last few turns go back to the model on the next one,
so "the incline one" makes sense, even after a restart. The model answers with a typed
choice; code writes the reply from this turn's tool results (replies.render). Before a
reply is shown, code also checks every number in it against the tool results the model
saw (this turn's, and those in the history it was given); a reply that fails is replaced
by a refusal and not saved.
A message the safety screen refuses gets the fixed refusal, with no model call; the turn
is saved without model messages. A failed or abandoned run saves nothing.
handle_agent_message is the same path for callers that only want the reply.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import asyncpg
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ToolReturnPart,
)
from pydantic_ai.usage import UsageLimits

from workout_agent.agents.guardrails import page_result
from workout_agent.agents.outputs import AgentOutput, ShowWorkoutDay
from workout_agent.agents.workout_agent import AgentDeps, jsonable
from workout_agent.observability import TurnTrace
from workout_agent.sessions.chat_store import load_history, save_turn, try_lock_chat, unlock_chat
from workout_agent.sessions.daily_checks import LATEST_RECORD, latest_record_today
from workout_agent.sessions.history import shrink_old_tool_results, tool_results_in
from workout_agent.sessions.titles import chat_title
from workout_agent.tools.find_exercise import Searcher
from workout_agent.tools.workout_day import planned_day
from workout_agent.workflow.events import ChatEvent, Error, Final, ToolEnd, ToolStart, tool_label
from workout_agent.workflow.replies import (
    UNVERIFIED_REPLY,
    RenderError,
    render,
    verify_agent_reply,
)
from workout_agent.workflow.safety import SafetyRefused

logger = logging.getLogger(__name__)

BUSY_REPLY = "This chat is busy answering another message. Please send yours again."
MAX_TOOL_CALLS = 6  # docs/amendments.md §9


def _tool_result(event: FunctionToolResultEvent) -> dict[str, Any]:
    """The tool's result dict (already through the tool contract), or a retry marker."""
    part = event.part
    if isinstance(part, ToolReturnPart) and isinstance(part.content, dict):
        return part.content
    return {"status": "retry"}  # the model sent bad arguments and is asked to try again


async def _run_agent(
    agent: Agent[AgentDeps, AgentOutput],
    text: str,
    deps: AgentDeps,
    history: list[ModelMessage],
    queue: asyncio.Queue[ChatEvent | None],
) -> AgentRunResult[AgentOutput]:
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
                            args = event.part.args_as_dict()
                            queue.put_nowait(
                                ToolStart(
                                    tool=tool,
                                    label=tool_label(tool, args),
                                    args=args,
                                    call_id=event.part.tool_call_id,
                                )
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            tool = event.part.tool_name or ""
                            result = _tool_result(event)
                            queue.put_nowait(
                                ToolEnd(
                                    tool=tool,
                                    status=str(result.get("status", "ok")),
                                    result=result,
                                    call_id=event.tool_call_id,
                                )
                            )
    finally:
        queue.put_nowait(None)
    if run.result is None:  # iter() always ends with a result unless it raised
        raise RuntimeError("agent run ended without a result")
    return run.result


async def stream_agent_message(
    conn: asyncpg.Connection,
    chat_id: str,
    text: str,
    agent: Agent[AgentDeps, AgentOutput],
    search: Searcher | None = None,
    history_turns: int = 10,
    full_tool_turns: int = 2,
) -> AsyncGenerator[ChatEvent]:
    # The trace for this turn; finish() records how it ended (observability.py).
    turn = TurnTrace(chat_id, text)
    outcome, final_reply, verified = "error", None, None
    try:
        # Turn a second message away before any work, not after the user watched it run.
        if not await try_lock_chat(conn, chat_id):
            outcome = "busy"
            yield Error(message=BUSY_REPLY)
            return
        try:
            history, last_turn = await load_history(conn, chat_id, history_turns)
            history = shrink_old_tool_results(history, full_tool_turns)
            deps = AgentDeps(conn=conn, search=search)  # fresh evidence list for each question
            # Once a day, the first message looks up how recent the history is; later chats
            # read the stored result. Stored and traced only: the model never sees it.
            try:
                latest, looked_up = await latest_record_today(conn, deps.today)
            except Exception:  # a freshness problem must not block the answer
                logger.exception("latest_record check failed for chat %s", chat_id)
            else:
                if looked_up:
                    yield ToolStart(
                        tool=LATEST_RECORD,
                        label=tool_label(LATEST_RECORD, {}),
                        call_id=LATEST_RECORD,
                    )
                    deps.evidence.append(latest)
                    turn.tool_span(LATEST_RECORD, {}, latest)
                    yield ToolEnd(
                        tool=LATEST_RECORD,
                        status=latest["status"],
                        result=page_result(LATEST_RECORD, latest),
                        call_id=LATEST_RECORD,
                    )
            # The run happens in its own task and hands events over through a queue: the
            # agent's internals must start and finish in one task, and this generator may
            # be closed early (the client went away), which then just cancels that task.
            # The task runs in the turn's trace context, so its spans nest under the turn.
            queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()
            task = asyncio.create_task(
                _run_agent(agent, text, deps, history, queue), context=turn.context
            )
            try:
                while (event := await queue.get()) is not None:
                    yield event
                result = await task  # raises if the run failed
            except SafetyRefused as refused:
                # The safety screen matched before any model call: the fixed refusal is the
                # reply. The turn is saved without model messages, so the chat shows it but
                # the model never sees the refused text.
                if await _save(conn, chat_id, last_turn + 1, text, refused.reply, [], []):
                    outcome, final_reply, verified = "refused", refused.reply, True
                    yield Final(reply=refused.reply, verified=True, checked=0)
                else:
                    outcome = "busy"
                    yield Error(message=BUSY_REPLY)
                return
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.wait({task})  # it must stop using conn before we go on

            # A chosen day: code fetches its exercises now. The long list never goes to the
            # model, but it is a tool result like any other: kept as evidence and checked.
            if isinstance(result.output, ShowWorkoutDay):
                day_args = {"day_id": result.output.day_id}
                yield ToolStart(
                    tool="planned_day",
                    label=tool_label("planned_day", day_args),
                    args=day_args,
                    call_id="planned_day",
                )
                planned = jsonable(await planned_day(conn, result.output.day_id))
                deps.evidence.append(planned)
                yield ToolEnd(
                    tool="planned_day",
                    status=planned["status"],
                    result=page_result("planned_day", planned),
                    call_id="planned_day",
                )

            # The model chose; code writes the words, only from this turn's tool results.
            # (The output validator already sent back any choice that cites what no tool
            # returned, so this failing means a bug; it is still never shown or saved.)
            try:
                reply = render(result.output, deps.evidence)
            except RenderError:
                outcome = "render_error"
                yield Error(message=UNVERIFIED_REPLY)
                return

            # Verification is code: every number must come from a tool result the model saw.
            seen = tool_results_in(history) + deps.evidence
            if not verify_agent_reply(reply, seen):
                outcome, final_reply, verified = "unverified", reply, False
                yield Final(reply=UNVERIFIED_REPLY, verified=False, checked=len(seen))
                return  # the request did not succeed, so nothing is saved

            # new_messages() is only this turn; older turns are already saved.
            new_messages = result.new_messages()
            if not await _save(
                conn, chat_id, last_turn + 1, text, reply, new_messages, deps.evidence
            ):
                outcome = "busy"
                yield Error(message=BUSY_REPLY)
                return
            outcome, final_reply, verified = type(result.output).__name__, reply, True
            yield Final(reply=reply, verified=True, checked=len(seen))
        finally:
            await unlock_chat(conn, chat_id)
    finally:
        turn.finish(outcome, final_reply, verified)


async def _save(
    conn: asyncpg.Connection,
    chat_id: str,
    turn_no: int,
    text: str,
    reply: str,
    messages: list[ModelMessage],
    evidence: list[dict[str, Any]],
) -> bool:
    """Save one answered turn. False if another message in this chat saved it first."""
    title, is_fallback = chat_title(text, evidence)
    try:
        await save_turn(conn, chat_id, turn_no, text, reply, messages, evidence, title, is_fallback)
    except asyncpg.UniqueViolationError:
        return False  # another message in this chat finished first; don't overwrite it
    return True


async def handle_agent_message(
    conn: asyncpg.Connection,
    chat_id: str,
    text: str,
    agent: Agent[AgentDeps, AgentOutput],
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
