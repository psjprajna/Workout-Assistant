"""How the chat page talks to the API: POST /chat for the reply, or /chat/stream for events.

The client carries the API's base URL, so tests can hand in a mocked or in-process one.
Failures come back as values (reply text, or an Error event), so the page always has
something to show.
"""

from collections.abc import AsyncIterator

import httpx

from workout_agent.workflow.events import ChatEvent, Error, chat_event_adapter


def _status_error(e: httpx.HTTPStatusError) -> str:
    return f"The API answered with an error ({e.response.status_code}). Please try again."


def _unreachable(client: httpx.AsyncClient) -> str:
    return f"Can't reach the API at {client.base_url}. Is uvicorn running?"


async def send_message(client: httpx.AsyncClient, chat_id: str, message: str) -> str:
    try:
        response = await client.post(
            "/chat", json={"chat_id": chat_id, "message": message}
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return _status_error(e)
    except httpx.RequestError:
        return _unreachable(client)
    return response.json()["reply"]


async def stream_message(
    client: httpx.AsyncClient, chat_id: str, message: str
) -> AsyncIterator[ChatEvent]:
    """The events of POST /chat/stream as they arrive; `: ping` lines are skipped."""
    try:
        async with client.stream(
            "POST", "/chat/stream", json={"chat_id": chat_id, "message": message}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield chat_event_adapter.validate_json(line.removeprefix("data: "))
    except httpx.HTTPStatusError as e:
        yield Error(message=_status_error(e))
    except httpx.RequestError:
        yield Error(message=_unreachable(client))
