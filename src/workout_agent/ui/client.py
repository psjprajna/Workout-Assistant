"""The one function the chat page calls: send a message to POST /chat, get the reply text.

The client carries the API's base URL, so tests can hand in a mocked or in-process one.
Failures come back as reply text, so the page always has something to show.
"""

import httpx


async def send_message(client: httpx.AsyncClient, chat_id: str, message: str) -> str:
    try:
        response = await client.post(
            "/chat", json={"chat_id": chat_id, "message": message}
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"The API answered with an error ({e.response.status_code}). Please try again."
    except httpx.RequestError:
        return f"Can't reach the API at {client.base_url}. Is uvicorn running?"
    return response.json()["reply"]
