"""Unit tests: send_message() against a mocked POST /chat, no server needed."""

import json

import httpx

from workout_agent.ui.client import send_message

API_URL = "http://api.test"


def _client():
    return httpx.AsyncClient(base_url=API_URL)


async def test_posts_chat_id_and_message_and_returns_the_reply(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chat", json={"reply": "Which one? 1. Bench Press"})

    async with _client() as client:
        reply = await send_message(client, "s1", "last time I did bench press?")

    assert reply == "Which one? 1. Bench Press"
    sent = httpx_mock.get_request()
    assert sent.method == "POST"
    assert json.loads(sent.content) == {
        "chat_id": "s1",
        "message": "last time I did bench press?",
    }


async def test_error_status_becomes_a_reply(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chat", status_code=500)

    async with _client() as client:
        reply = await send_message(client, "s1", "hi")

    assert "500" in reply


async def test_unreachable_api_becomes_a_reply(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("refused"))

    async with _client() as client:
        reply = await send_message(client, "s1", "hi")

    assert API_URL in reply
    assert "uvicorn" in reply
