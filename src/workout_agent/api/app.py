"""The HTTP API (FastAPI): POST /chat takes a chat id and a message, returns the reply.

create_app wires up a database connection pool and the router to use. With an agent,
each chat's turns are saved in Postgres, and GET/PATCH/DELETE /chats list, rename and
remove them (the chat page's sidebar). Given a Qdrant URL and an embedder, the agent's
find_exercise also searches by meaning. GET /health is a simple "is it running" check.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial

import asyncpg
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from qdrant_client import AsyncQdrantClient

from workout_agent.agents.workout_agent import AgentDeps
from workout_agent.db.migrate import apply_migrations
from workout_agent.embeddings.index import Embedder
from workout_agent.embeddings.search import search_exercises
from workout_agent.sessions import chat_store
from workout_agent.workflow.agent_chat import handle_agent_message
from workout_agent.workflow.chat import Route, Session, handle_message


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
    agent: Agent[AgentDeps, str] | None = None,
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
        if qdrant_url and embedder:
            qdrant = AsyncQdrantClient(url=qdrant_url)
            app.state.search = partial(search_exercises, qdrant, embedder)
        yield
        # Shutdown: close the pool and the Qdrant client cleanly.
        await app.state.pool.close()
        if qdrant is not None:
            await qdrant.close()

    app = FastAPI(title="Workout Agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
