"""The HTTP API (FastAPI): POST /chat takes a chat id and a message, returns the reply.

POST /chat/stream takes the same body and answers with server-sent events: one
`data: {json}` line per tool start/end, then the reply (workflow/events.py), and a
`: ping` comment while nothing else is happening so proxies keep the connection open.

create_app wires up a database connection pool and the router to use. With an agent,
each chat's turns are saved in Postgres, and GET/PATCH/DELETE /chats list, rename and
remove them (the chat page's sidebar). Given a Qdrant URL and an embedder, the agent's
find_exercise also searches by meaning. GET /health is a simple "is it running" check.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime
from functools import partial
from typing import Any

import asyncpg
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from qdrant_client import AsyncQdrantClient

from workout_agent.agents.outputs import AgentOutput
from workout_agent.agents.workout_agent import AgentDeps
from workout_agent.db.migrate import apply_migrations
from workout_agent.embeddings.index import Embedder
from workout_agent.embeddings.search import search_exercises
from workout_agent.importing.upload import ImportResult, import_export
from workout_agent.observability import flush_tracing
from workout_agent.sessions import chat_store, daily_checks
from workout_agent.workflow.agent_chat import handle_agent_message, stream_agent_message
from workout_agent.workflow.chat import Route, Session, handle_message
from workout_agent.workflow.events import ChatEvent, Error, Final, chat_event_adapter

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 5.0
STREAM_FAILED = "Something went wrong while answering. Please try again."


class ChatRequest(BaseModel):
    chat_id: str = Field(min_length=1)  # the chat page's thread id
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


class ChatSummaryOut(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class TurnOut(BaseModel):
    user_text: str
    reply_text: str
    created_at: datetime


class ChatOut(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    turns: list[TurnOut]


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def create_app(
    route: Route,
    database_url: str,
    agent: Agent[AgentDeps, AgentOutput] | None = None,
    qdrant_url: str | None = None,
    embedder: Embedder | None = None,
    history_turns: int = 10,
    history_full_tool_turns: int = 2,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: a pool of ready database connections, shared by all requests.
        app.state.pool = await asyncpg.create_pool(database_url)
        # Every migration is IF NOT EXISTS, so this only adds what's new (e.g. chat history).
        async with app.state.pool.acquire() as conn:
            await apply_migrations(conn)
        # Rule-router conversations only (used when no agent is given); not persisted.
        app.state.sessions = {}
        # Search by meaning only when both Qdrant and an embedder were given.
        qdrant = None
        app.state.search = None
        app.state.qdrant = None
        app.state.embedder = embedder
        if qdrant_url and embedder:
            qdrant = AsyncQdrantClient(url=qdrant_url)
            app.state.search = partial(search_exercises, qdrant, embedder)
            app.state.qdrant = qdrant
        yield
        # Shutdown: close the pool and the Qdrant client cleanly; send any buffered traces.
        await app.state.pool.close()
        if qdrant is not None:
            await qdrant.close()
        flush_tracing()

    app = FastAPI(title="Workout Agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/data/prompt")
    async def data_prompt() -> dict[str, Any]:
        """Should the chat page ask for a newer Jefit export? True once a day, at most."""
        async with app.state.pool.acquire() as conn:
            ask, latest_on = await daily_checks.ask_for_export_today(conn, date.today())
        return {"ask": ask, "latest_on": latest_on.isoformat() if latest_on else None}

    @app.post("/data/import")
    async def data_import(request: Request) -> dict[str, Any]:
        """Import the Jefit export sent as the request body (the file's text)."""
        try:
            text = (await request.body()).decode("utf-8-sig")
        except UnicodeDecodeError:
            return asdict(ImportResult("rejected", "the file is not text"))
        async with app.state.pool.acquire() as conn:
            result = await import_export(conn, text, app.state.qdrant, app.state.embedder)
            if result.status == "ok":  # the freshness check should see the new data
                await daily_checks.forget_latest_record(conn, date.today())
        return asdict(result)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if agent is not None:
            async with app.state.pool.acquire() as conn:
                reply = await handle_agent_message(
                    conn,
                    request.chat_id,
                    request.message,
                    agent,
                    search=app.state.search,
                    history_turns=history_turns,
                    full_tool_turns=history_full_tool_turns,
                )
            return ChatResponse(reply=reply)
        # Each chat id gets its own Session, so pending questions stay separate.
        sessions: dict[str, Session] = app.state.sessions
        session = sessions.setdefault(request.chat_id, Session())

        # Borrow a connection for this request and give it back afterwards.
        async with app.state.pool.acquire() as conn:
            reply = await handle_message(conn, session, request.message, route)

        return ChatResponse(reply=reply)

    async def chat_events(request: ChatRequest) -> AsyncIterator[ChatEvent]:
        # The connection is borrowed here, not in the route: the route returns before
        # the stream ends, and the connection must stay ours until the last event.
        async with app.state.pool.acquire() as conn:
            if agent is not None:
                async for event in stream_agent_message(
                    conn,
                    request.chat_id,
                    request.message,
                    agent,
                    search=app.state.search,
                    history_turns=history_turns,
                    full_tool_turns=history_full_tool_turns,
                ):
                    yield event
            else:
                session = app.state.sessions.setdefault(request.chat_id, Session())
                yield Final(reply=await handle_message(conn, session, request.message, route))

    async def sse(request: ChatRequest) -> AsyncIterator[str]:
        """Events as SSE lines, with a ping whenever none came for HEARTBEAT_SECONDS."""
        queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for event in chat_events(request):
                    await queue.put(event)
            except Exception:
                logger.exception("chat stream failed for chat %s", request.chat_id)
                await queue.put(Error(message=STREAM_FAILED))
            finally:
                await queue.put(None)  # end of stream

        task = asyncio.create_task(produce())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                if event is None:
                    break
                yield f"data: {chat_event_adapter.dump_json(event).decode()}\n\n"
        finally:
            # The client went away (or we finished): stop the run; nothing is saved.
            task.cancel()

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        return StreamingResponse(sse(request), media_type="text/event-stream")

    @app.get("/chats", response_model=list[ChatSummaryOut])
    async def list_chats(limit: int = 20, before: datetime | None = None) -> list[ChatSummaryOut]:
        async with app.state.pool.acquire() as conn:
            chats = await chat_store.list_chats(conn, limit, before)
        return [ChatSummaryOut.model_validate(c, from_attributes=True) for c in chats]

    @app.get("/chats/{chat_id}", response_model=ChatOut)
    async def get_chat(chat_id: str) -> ChatOut:
        async with app.state.pool.acquire() as conn:
            chat = await chat_store.get_chat(conn, chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="chat not found")
        return ChatOut.model_validate(chat, from_attributes=True)

    @app.delete("/chats/{chat_id}", status_code=204)
    async def delete_chat(chat_id: str) -> Response:
        async with app.state.pool.acquire() as conn:
            await chat_store.delete_chat(conn, chat_id)
        return Response(status_code=204)

    @app.patch("/chats/{chat_id}", status_code=204)
    async def rename_chat(chat_id: str, request: RenameRequest) -> Response:
        async with app.state.pool.acquire() as conn:
            found = await chat_store.rename_chat(conn, chat_id, request.title)
        if not found:
            raise HTTPException(status_code=404, detail="chat not found")
        return Response(status_code=204)

    return app
