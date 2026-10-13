"""What the chat stream sends while the agent works: tool progress, then the reply.

Each event is one line on the wire (`data: {json}`), so the page can show "Searching
your exercises…" while the slow path runs. Labels are written here, from the tool name
and its arguments, never by the model.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ToolStart(BaseModel):
    kind: Literal["tool_start"] = "tool_start"
    tool: str
    label: str


class ToolEnd(BaseModel):
    kind: Literal["tool_end"] = "tool_end"
    tool: str
    status: str  # the tool result's status: ok / ambiguous / no_rows, or "error"


class Final(BaseModel):
    kind: Literal["final"] = "final"
    reply: str
    # True when code checked every number in the reply against `checked` tool results.
    verified: bool = False
    checked: int = 0


class Error(BaseModel):
    kind: Literal["error"] = "error"
    message: str


ChatEvent = Annotated[ToolStart | ToolEnd | Final | Error, Field(discriminator="kind")]
chat_event_adapter: TypeAdapter[ChatEvent] = TypeAdapter(ChatEvent)


def tool_label(tool: str, args: dict[str, Any]) -> str:
    if tool == "find_exercise_tool":
        return f"Searching your exercises for '{args.get('query', '')}'…"
    if tool == "get_last_session_tool":
        return "Fetching your last session…"
    return f"Running {tool}…"
