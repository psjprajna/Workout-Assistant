"""Integration tests: POST /chat and GET /health through the real FastAPI app."""

import httpx
import pytest

from workout_agent.api.app import create_app
from workout_agent.workflow.simple_router import simple_route


@pytest.fixture
async def client(seeded, test_database_url):
    app = create_app(route=simple_route, database_url=test_database_url)
    # httpx does not run the app's startup/shutdown, so we do it ourselves
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def _chat(client, session_id, message):
    return await client.post("/chat", json={"session_id": session_id, "message": message})


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_chat_asks_which_one(client):
    r = await _chat(client, "a", "last time I did bench press?")
    assert r.status_code == 200
    assert "Which one" in r.json()["reply"]


async def test_same_session_remembers_the_question(client):
    await _chat(client, "a", "last time I did bench press?")
    r = await _chat(client, "a", "1")
    assert "kg" in r.json()["reply"]


async def test_sessions_do_not_share_questions(client):
    await _chat(client, "a", "last time I did bench press?")
    r = await _chat(client, "b", "1")
    assert "kg" not in r.json()["reply"]


async def test_empty_message_is_rejected(client):
    r = await _chat(client, "a", "")
    assert r.status_code == 422


async def test_missing_session_id_is_rejected(client):
    r = await client.post("/chat", json={"message": "hi"})
    assert r.status_code == 422

@pytest.mark.llm
async def test_chat_with_real_agent_remembers_the_question(seeded, test_database_url):
    from workout_agent.agents.workout_agent import build_agent
    from workout_agent.settings import Settings

    app = create_app(route=simple_route, database_url=test_database_url,
                     agent=build_agent(Settings()))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            first = await _chat(c, "s1", "last time I did bench press?")
            assert "incline" in first.json()["reply"].lower()  # it asks which one

            second = await _chat(c, "s1", "the incline one")
            assert "60" in second.json()["reply"]  # incline's seeded weight

async def test_chat_page_function_talks_to_the_real_endpoint(client):
    from workout_agent.ui.client import send_message

    reply = await send_message(client, "page", "last time I did bench press?")
    assert "Which one" in reply
