"""The chat page: uv run chainlit run src/workout_agent/ui/chat.py --port 8001

Every message goes to the running API's POST /chat/stream (start uvicorn first), so the
page, the API and the agent are all in the path. Each tool call shows as a step while it
runs ("Searching your exercises for 'bench press'…"), then the reply arrives whole.
Each chat is a Chainlit thread; its id is the chat id the API saves turns under, so past
chats show in the sidebar and can be reopened (ui/data_layer.py). There is one user,
logged in automatically: no login form.
Chainlit reads CHAINLIT_AUTH_SECRET from .env; create it with `uv run chainlit create-secret`.
"""

import chainlit as cl
import httpx
from chainlit.utils import utc_now
from starlette.datastructures import Headers

from workout_agent.sessions.titles import chat_title
from workout_agent.settings import Settings
from workout_agent.ui.client import stream_message
from workout_agent.ui.data_layer import USER, ApiDataLayer
from workout_agent.workflow.events import Error, Final, ToolEnd, ToolStart

_STATUS_TEXT = {"ok": "✓ ok"}


def _new_client() -> httpx.AsyncClient:
    # Agent turns with several tool calls can take longer than httpx's 5 s default.
    return httpx.AsyncClient(base_url=Settings().api_url, timeout=60)


# Chainlit creates the data layer once per process.
_data_layer = ApiDataLayer(_new_client())


@cl.data_layer
def data_layer() -> ApiDataLayer:
    return _data_layer


@cl.header_auth_callback
async def auth(headers: Headers) -> cl.User:
    return cl.User(identifier=USER)  # always succeeds, so no login form is shown


@cl.set_starters
async def starters(user: cl.User | None = None) -> list[cl.Starter]:
    return [
        cl.Starter(label="Last bench press", message="What did I lift last time on bench press?"),
        cl.Starter(label="Last barbell curl", message="When did I last do barbell curls?"),
        cl.Starter(label="Something for back", message="What back exercises have I done?"),
    ]


@cl.on_chat_start
async def start() -> None:
    # No welcome message: a bot message would save an empty chat to the sidebar.
    cl.user_session.set("client", _new_client())


@cl.on_chat_resume
async def resume(thread: cl.types.ThreadDict) -> None:
    # Chainlit shows the old messages from the data layer; the API has the model history.
    cl.user_session.set("client", _new_client())


@cl.on_message
async def on_message(message: cl.Message) -> None:
    client: httpx.AsyncClient = cl.user_session.get("client")  # type: ignore[assignment]
    # thread_id stays the same when a chat is reopened; session.id changes per connection.
    thread_id = cl.context.session.thread_id
    _data_layer.mark_pending(thread_id, chat_title(message.content, [])[0])
    open_steps: list[tuple[str, cl.Step]] = []  # in call order, closed as results arrive
    # Steps made without `async with` get no parent; nest them in this message's run, as
    # the reply is, so they show above the reply rather than after it.
    run = cl.context.current_step
    reply = ""
    try:
        async for event in stream_message(client, thread_id, message.content):
            if isinstance(event, ToolStart):
                step = cl.Step(
                    name=event.label,
                    type="tool",
                    show_input=False,
                    parent_id=run.id if run else None,
                )
                step.start = utc_now()  # start without end: the page shows it as running
                await step.send()
                open_steps.append((event.tool, step))
            elif isinstance(event, ToolEnd):
                found = next((i for i, (t, _) in enumerate(open_steps) if t == event.tool), None)
                if found is not None:
                    _, step = open_steps.pop(found)
                    step.output = _STATUS_TEXT.get(event.status, event.status)
                    step.end = utc_now()
                    await step.update()
            elif isinstance(event, Final):
                reply = event.reply
            elif isinstance(event, Error):
                reply = event.message
    finally:
        _data_layer.clear_pending(thread_id)
    await cl.Message(content=reply or "No answer came back. Please try again.").send()


@cl.on_chat_end
async def end() -> None:
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    if client is not None:
        await client.aclose()
