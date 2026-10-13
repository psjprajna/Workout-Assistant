"""Unit tests: send_message() and stream_message() against a mocked API, no server needed."""

import json

import httpx

from workout_agent.ui.client import send_message, stream_message
from workout_agent.workflow.events import Error, Final, ToolEnd, ToolStart

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


SSE_BODY = (
    'data: {"kind":"tool_start","tool":"find_exercise_tool","label":"Searching…"}\n\n'
    ": ping\n\n"
    'data: {"kind":"tool_end","tool":"find_exercise_tool","status":"ok"}\n\n'
    'data: {"kind":"final","reply":"80.00 kg"}\n\n'
)


async def _events(client):
    return [e async for e in stream_message(client, "s1", "bench?")]


async def test_stream_yields_events_and_skips_pings(httpx_mock):
    httpx_mock.add_response(
        url=f"{API_URL}/chat/stream",
        content=SSE_BODY.encode(),
        headers={"content-type": "text/event-stream"},
    )

    async with _client() as client:
        events = await _events(client)

    assert events == [
        ToolStart(tool="find_exercise_tool", label="Searching…"),
        ToolEnd(tool="find_exercise_tool", status="ok"),
        Final(reply="80.00 kg"),
    ]
    assert json.loads(httpx_mock.get_request().content) == {"chat_id": "s1", "message": "bench?"}


async def test_stream_error_status_becomes_an_error_event(httpx_mock):
    httpx_mock.add_response(url=f"{API_URL}/chat/stream", status_code=500)

    async with _client() as client:
        events = await _events(client)

    assert len(events) == 1
    assert isinstance(events[0], Error)
    assert "500" in events[0].message


async def test_stream_unreachable_api_becomes_an_error_event(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("refused"))

    async with _client() as client:
        events = await _events(client)

    assert len(events) == 1
    assert isinstance(events[0], Error)
    assert "uvicorn" in events[0].message
