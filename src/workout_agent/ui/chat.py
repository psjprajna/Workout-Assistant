"""The chat page: uv run chainlit run src/workout_agent/ui/chat.py --port 8001

Every message goes to the running API's POST /chat (start uvicorn first), so the page,
the API and the agent are all in the path. Each browser chat uses its own session id.
"""

import chainlit as cl
import httpx

from workout_agent.settings import Settings
from workout_agent.ui.client import send_message


@cl.on_chat_start
async def start() -> None:
    # Agent turns with several tool calls can take longer than httpx's 5 s default.
    cl.user_session.set("client", httpx.AsyncClient(base_url=Settings().api_url, timeout=60))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    client: httpx.AsyncClient = cl.user_session.get("client")  # type: ignore[assignment]
    reply = await send_message(client, cl.context.session.id, message.content)
    await cl.Message(content=reply).send()


@cl.on_chat_end
async def end() -> None:
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    if client is not None:
        await client.aclose()
