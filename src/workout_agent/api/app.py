"""The HTTP API (FastAPI): POST /chat takes a session id and a message, returns the reply.

create_app wires up a database connection pool, one in-memory Session per session id,
and the router to use. Given a Qdrant URL and an embedder, the agent's find_exercise
also searches by meaning. GET /health is a simple "is it running" check.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

import asyncpg
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from qdrant_client import AsyncQdrantClient

from workout_agent.agents.workout_agent import AgentDeps
from workout_agent.embeddings.index import Embedder
from workout_agent.embeddings.search import search_exercises
from workout_agent.workflow.agent_chat import AgentSession, handle_agent_message
from workout_agent.workflow.chat import Route, Session, handle_message


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


def create_app(
    route: Route,
    database_url: str,
    agent: Agent[AgentDeps, str] | None = None,
    qdrant_url: str | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: a pool of ready database connections, shared by all requests,
        # and an in-memory dictionary of conversations keyed by session id.
        app.state.pool = await asyncpg.create_pool(database_url)
        app.state.sessions = {}
        app.state.agent_sessions = {}
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
            agent_session = app.state.agent_sessions.setdefault(
                request.session_id, AgentSession()
            )
            async with app.state.pool.acquire() as conn:
                reply = await handle_agent_message(
                    conn, agent_session, request.message, agent, search=app.state.search
                )
            return ChatResponse(reply=reply)
        # Each session id gets its own Session, so pending questions stay separate.
        sessions: dict[str, Session] = app.state.sessions
        session = sessions.setdefault(request.session_id, Session())

        # Borrow a connection for this request and give it back afterwards.
        async with app.state.pool.acquire() as conn:
            reply = await handle_message(conn, session, request.message, route)

        return ChatResponse(reply=reply)

    return app